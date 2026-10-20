import random
from itertools import permutations
import torch
import torch.nn as nn
import torch.nn.functional as F

from velocity_predictor_model import PhaseCorrelation


class MEConvLSTMCell(nn.Module):
    # velocities are whole pixels, so warp() can use an exact integer shift
    integer_shift = True

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=padding,
            padding_mode="circular", bias=bias
        )

        if bias:
            nn.init.constant_(self.conv.bias[hidden_dim:2 * hidden_dim], 1.0)

        self._meshgrid_cache = {}

    def _get_base_grid(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        cached = self._meshgrid_cache.get(key)
        if cached is None:
            cached = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij"
            )
            self._meshgrid_cache[key] = cached
        return cached

    def warp(self, x, u):
        B, K, C, H, W = x.shape
        x = x.reshape(B * K, C, H, W)
        u = u.reshape(B * K, 2)

        dx = u[:, 0, None, None]
        dy = u[:, 1, None, None]

        yy, xx = self._get_base_grid(H, W, x.device, x.dtype)
        yy = yy.unsqueeze(0).expand(B * K, -1, -1)
        xx = xx.unsqueeze(0).expand(B * K, -1, -1)

        if self.integer_shift:
            yy = 2 * torch.remainder(yy - dy, H) / (H - 1) - 1
            xx = 2 * torch.remainder(xx - dx, W) / (W - 1) - 1
        else:
            # 1-px circular pad: fractional coordinates near the wrap must interpolate across it
            x = F.pad(x, (1, 1, 1, 1), mode="circular")

            yy = 2 * (torch.remainder(yy - dy, H) + 1) / (H + 1) - 1
            xx = 2 * (torch.remainder(xx - dx, W) + 1) / (W + 1) - 1

        grid = torch.stack([xx, yy], dim=-1)
        x = F.grid_sample(x, grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
        return x.view(B, K, C, H, W)

    def forward(self, x, h, c, u):
        B, K, Ch, H, W = h.shape

        h = self.warp(h, u)
        c = self.warp(c, u)

        x_exp = x.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, -1, H, W)
        h     = h.reshape(B * K, Ch, H, W)
        c     = c.reshape(B * K, Ch, H, W)

        i, f, o, g = torch.chunk(
            self.conv(torch.cat([x_exp, h], dim=1)), 4, dim=1
        )
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h.view(B, K, Ch, H, W), c.view(B, K, Ch, H, W)

    def init_hidden(self, B, K, H, W, device, dtype):
        z = torch.zeros(B, K, self.hidden_dim, H, W, device=device, dtype=dtype)
        return z, torch.zeros_like(z)


class Seq2SeqMEConvLSTM(nn.Module):
    """MELSTM video predictor: encoder-decoder with K velocity slots tracked by phase correlation."""

    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels=None,
                 n_slots=2,
                 kernel_size=3,
                 slot_reduce='max',
                 decoder_layers=1,
                 decoder_channels=None,
                 bias=True,
                 batch_first=True,
                 phase_corr_kwargs=None,
                 track_mode="h",
                 track_prior_weight=0.5,
                 n_proposals=None,
                 track_window=2,
                 detach_feedback=True,
                 decoder_act="relu",
                 decoder_skip=False):
        super().__init__()
        if track_mode not in ("h", "propose", "prior", "window", "tie", "propose_distinct"):
            raise ValueError(f"unknown track_mode {track_mode!r}")
        if decoder_act not in ("relu", "leaky"):
            raise ValueError(f"decoder_act {decoder_act!r}: expected 'relu' or 'leaky'")
        self.track_mode         = track_mode
        self.track_prior_weight = track_prior_weight
        self.n_proposals        = n_proposals or n_slots
        self.track_window       = track_window
        self.detach_feedback    = detach_feedback
        self.x_track_until      = 0
        self.x_track_p          = None
        self.frozen_prob        = 0.0

        self.batch_first     = batch_first
        self.n_slots         = n_slots
        self.hidden_channels = hidden_channels
        self.slot_reduce     = slot_reduce
        output_channels      = output_channels or input_channels
        decoder_channels     = decoder_channels or hidden_channels
        self.decoder_channels = decoder_channels

        pc_kw = phase_corr_kwargs or {}

        self.phase_corr_bootstrap = PhaseCorrelation(n_modes=n_slots, **pc_kw)
        self.phase_corr_track     = PhaseCorrelation(n_modes=1,       **pc_kw)

        self.cell = MEConvLSTMCell(input_channels, hidden_channels,
                                   kernel_size, bias)

        layers = []
        in_ch = hidden_channels
        for _ in range(decoder_layers):
            layers += [nn.Conv2d(in_ch, decoder_channels,
                                 3, padding=1, padding_mode='circular', bias=True),
                       nn.LeakyReLU(0.01) if decoder_act == "leaky" else nn.ReLU()]
            in_ch = decoder_channels
        layers += [nn.Conv2d(in_ch, output_channels,
                             3, padding=1, padding_mode='circular', bias=True)]
        self.decoder = nn.Sequential(*layers)
        self.decoder_skip = (nn.Conv2d(hidden_channels, output_channels, 1, bias=True)
                             if decoder_skip else None)

    def decode(self, pooled):
        """Decoder output for the slot-pooled state (plus the linear skip, if on)."""
        pred = self.decoder(pooled)
        if self.decoder_skip is not None:
            pred = pred + self.decoder_skip(pooled)
        return pred

    def bootstrap_velocities(self, x0, x1):
        """K peaks from raw frame pair. Called once: encoder t=1."""
        v, _ = self.phase_corr_bootstrap(x0, x1)
        return v

    def _pc_surface(self, a, b):
        """Normalized phase-correlation surface of a against b, (..., H, W)."""
        H, W = a.shape[-2:]
        R = torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(b))
        R = R / (R.abs() + self.phase_corr_track.eps)
        return torch.fft.irfft2(R, s=(H, W))

    @staticmethod
    def _index_to_velocity(idx, H, W, dtype):
        """Flat surface index -> (vx, vy), exactly as PhaseCorrelation converts it."""
        y = torch.div(idx, W, rounding_mode="floor").to(dtype)
        x = (idx % W).to(dtype)
        x = torch.where(x > W / 2, x - W, x)
        y = torch.where(y > H / 2, y - H, y)
        return torch.stack((-x, -y), dim=-1)

    def track_velocities(self, h, frame, prev_frame=None):
        """Per-slot self-tracking: correlate each slot's h with frame. -> (B, K, 2)"""
        if self.track_mode in ("window", "tie"):
            return self._track_h_constrained(h, frame)
        if self.track_mode != "h" and prev_frame is not None:
            return self._track_with_frames(h, frame, prev_frame)
        B, K, Ch, H, W = h.shape
        _, C, _, _     = frame.shape

        h_tmpl = h.mean(dim=2).reshape(B * K, 1, H, W)
        f_rep  = (frame.unsqueeze(1)
                       .expand(B, K, C, H, W)
                       .reshape(B * K, C, H, W))

        with torch.no_grad():
            v_flat, _ = self.phase_corr_track(h_tmpl, f_rep)
        return v_flat.squeeze(1).reshape(B, K, 2)

    def _track_with_frames(self, h, frame, prev_frame):
        """track_mode "propose" / "prior"."""
        B, K, Ch, H, W = h.shape
        with torch.no_grad():
            f   = frame.mean(dim=1)
            tm  = h.mean(dim=2)
            s_h = self._pc_surface(tm, f.unsqueeze(1).expand_as(tm)).reshape(B, K, H * W)
            s_x = self._pc_surface(prev_frame.mean(dim=1), f).reshape(B, H * W)
            if self.track_mode == "propose":
                cand  = s_x.topk(self.n_proposals, dim=-1).indices
                cand  = cand.unsqueeze(1).expand(B, K, -1)
                score = torch.gather(s_h, 2, cand)
                idx   = torch.gather(cand, 2, score.argmax(-1, keepdim=True)).squeeze(-1)
            elif self.track_mode == "propose_distinct":
                cand  = s_x.topk(self.n_proposals, dim=-1).indices
                score = torch.gather(s_h, 2, cand.unsqueeze(1).expand(B, K, -1))
                perms = list(permutations(range(cand.shape[1]), K))
                total = torch.stack([sum(score[:, k, p[k]] for k in range(K)) for p in perms], 1)
                best  = torch.tensor(perms, device=h.device)[total.argmax(1)]
                idx   = torch.gather(cand, 1, best)
            else:
                nh  = s_h / s_h.amax(-1, keepdim=True).clamp_min(1e-8)
                nx  = s_x / s_x.amax(-1, keepdim=True).clamp_min(1e-8)
                idx = (nh + self.track_prior_weight * nx.unsqueeze(1)).argmax(-1)
            return self._index_to_velocity(idx, H, W, h.dtype)

    def _track_h_constrained(self, h, frame):
        """track_mode "window" / "tie": the h surfaces with a constraint."""
        B, K, Ch, H, W = h.shape
        with torch.no_grad():
            f   = frame.mean(dim=1)
            tm  = h.mean(dim=2)
            s_h = self._pc_surface(tm, f.unsqueeze(1).expand_as(tm)).reshape(B, K, H * W)
            if self.track_mode == "window":
                v_all = self._index_to_velocity(torch.arange(H * W, device=h.device), H, W, h.dtype)
                inwin = (v_all.abs() <= self.track_window).all(-1)
                idx = s_h.masked_fill(~inwin, float("-inf")).argmax(-1)
            else:
                order = s_h.amax(-1).argsort(dim=1, descending=True)
                taken = torch.zeros(B, H * W, dtype=torch.bool, device=h.device)
                idx   = torch.empty(B, K, dtype=torch.long, device=h.device)
                rows  = torch.arange(B, device=h.device)
                for r in range(K):
                    k  = order[:, r]
                    ix = s_h[rows, k].masked_fill(taken, float("-inf")).argmax(-1)
                    idx[rows, k] = ix
                    taken[rows, ix] = True
            return self._index_to_velocity(idx, H, W, h.dtype)

    def _track_x_continuity(self, prev_frame, frame, v_prev):
        """Top peaks of PC(x_{t-1}, x_t), each slot given the candidate closest to its previous velocity."""
        with torch.no_grad():
            cand, _ = self.phase_corr_bootstrap(prev_frame, frame)
            K = cand.shape[1]
            perms = list(permutations(range(K)))
            cost = torch.stack([(cand[:, list(p)] - v_prev).pow(2).sum((1, 2)) for p in perms])
            best = torch.tensor(perms, device=cand.device)[cost.argmin(0)]
            return torch.gather(cand, 1, best.unsqueeze(-1).expand(-1, -1, 2))

    def pool_slots(self, h):
        """h : (B, K, Ch, H, W) -> (B, Ch, H, W)"""
        if self.slot_reduce == 'mean':
            return h.mean(dim=1)
        elif self.slot_reduce == 'sum':
            return h.sum(dim=1)
        else:
            return h.max(dim=1).values

    def encode(self, input_seq, return_states=False):
        """Encoder only: (h, c, v_last, estimated_velocities, h_states), h and c the full per-slot state."""
        B, T_in, C, H, W = input_seq.shape
        K = self.n_slots

        h, c = self.cell.init_hidden(B, K, H, W, input_seq.device, input_seq.dtype)

        estimated_velocities = []
        h_states = [] if return_states else None
        v = torch.zeros(B, K, 2, device=input_seq.device, dtype=input_seq.dtype)

        for t in range(T_in):
            if t == 0:
                # t = 0: h = 0, so the warp is a no-op
                v = torch.zeros(B, K, 2, device=input_seq.device,
                                         dtype=input_seq.dtype)
            elif t == 1:
                # bootstrap slot velocities from (X_0, X_1)
                v = self.bootstrap_velocities(input_seq[:, 0], input_seq[:, 1])
            elif self.training and t < self.x_track_until:
                # training only: x-tracking curriculum (raw frame pair)
                v = self._track_x_continuity(input_seq[:, t - 1], input_seq[:, t], v)
            elif self.training and self.x_track_p is not None and self.x_track_p[t] > 0:
                # training only: stochastic handover between frame-pair and h-tracked velocities
                p_t = float(self.x_track_p[t])
                v_x = self._track_x_continuity(input_seq[:, t - 1], input_seq[:, t], v)
                if p_t >= 1:
                    v = v_x
                else:
                    v_h = self.track_velocities(h, input_seq[:, t],
                                                prev_frame=input_seq[:, t - 1])
                    use_x = torch.rand(B, device=input_seq.device) < p_t
                    v = torch.where(use_x.view(B, 1, 1), v_x, v_h)
            else:
                # slot self-tracking
                v = self.track_velocities(h, input_seq[:, t],
                                          prev_frame=input_seq[:, t - 1])

            h, c = self.cell(input_seq[:, t], h, c, v)

            if t > 0:
                estimated_velocities.append(v.detach())

            if return_states:
                h_states.append(h.mean(dim=2).detach())

        return h, c, v, estimated_velocities, h_states

    def forward(self,
                input_seq,
                pred_len,
                target_seq=None,
                track_decoder_velocity=True,
                return_velocity=False,
                return_states=False):
        """track_decoder_velocity: track against the true next frame (training); else freeze the last encoder velocity."""
        if not self.batch_first:
            input_seq = input_seq.permute(1, 0, 2, 3, 4)

        B, T_in, C, H, W = input_seq.shape
        K = self.n_slots

        h, c, v, estimated_velocities, h_states = self.encode(
            input_seq, return_states=return_states)

        prev_frame = input_seq[:, -1]
        outputs    = []
        v_enc_last = v
        freeze = None
        if (self.training and self.frozen_prob > 0 and target_seq is not None
                and track_decoder_velocity):
            freeze = (torch.rand(B, device=input_seq.device) < self.frozen_prob).view(B, 1, 1)

        for t in range(pred_len):
            current_frame = prev_frame.detach() if self.detach_feedback else prev_frame

            if target_seq is not None and track_decoder_velocity:
                true_prev = input_seq[:, -1] if t == 0 else target_seq[:, t - 1]
                v = self.track_velocities(h, target_seq[:, t], prev_frame=true_prev)
                if freeze is not None:
                    v = torch.where(freeze, v_enc_last, v)
                estimated_velocities.append(v.clone().detach())
            # frozen rollout: v keeps the last encoder estimate

            h, c  = self.cell(current_frame, h, c, v)
            pred  = self.decode(self.pool_slots(h))
            outputs.append(pred)
            prev_frame = pred

            if return_states:
                h_states.append(h.mean(dim=2).detach())

        outputs = torch.stack(outputs, dim=1)

        result = [outputs]

        if return_velocity:
            result.append(torch.stack(estimated_velocities, dim=1))

        if return_states:
            result.append({
                "h": torch.stack(h_states, dim=1),
            })

        if len(result) == 1:
            return result[0]
        return tuple(result)
