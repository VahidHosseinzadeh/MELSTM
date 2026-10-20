import numpy as np
import torch
import torch.nn as nn

from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM
from velocity_predictor_model import PhaseCorrelation


class VelocityPool(nn.Module):
    @staticmethod
    def output_channels(mode, n_velocities, hidden_channels):
        return hidden_channels * (n_velocities if mode == "concat" else 1)

    def __init__(self, mode, n_velocities, hidden_channels, score_hidden=32,
                 temperature=1.0):
        super().__init__()
        if mode not in ("attention", "max", "mean", "concat"):
            raise ValueError(f"unknown velocity_pool {mode!r}")
        self.mode = mode
        self.n_velocities = n_velocities
        self.temperature = temperature
        self.out_channels = self.output_channels(mode, n_velocities, hidden_channels)

        if mode == "attention":
            self.score = nn.Sequential(
                nn.Linear(2 * hidden_channels, score_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(score_hidden, 1),
            )

    def forward(self, h):
        if self.mode == "max":
            return h.amax(dim=1), None
        if self.mode == "mean":
            return h.mean(dim=1), None
        if self.mode == "concat":
            B, V, C, H, W = h.shape
            return h.reshape(B, V * C, H, W), None

        mu = h.mean(dim=(-2, -1))
        sd = h.std(dim=(-2, -1))
        w = torch.softmax(
            self.score(torch.cat([mu, sd], dim=-1)) / self.temperature, dim=1)
        return (h * w[..., None, None]).sum(dim=1), w.squeeze(-1)


class ConvClassifierHead(nn.Module):
    def __init__(self, in_channels, n_classes=10, channels=64, n_blocks=3,
                 mlp_hidden=128, dropout=0.0, norm="batch", groups=8, mlp_act="relu"):
        super().__init__()
        if norm not in ("group", "batch", "none"):
            raise ValueError(f"unknown head norm {norm!r}")
        if mlp_act not in ("relu", "leaky"):
            raise ValueError(f"unknown head mlp_act {mlp_act!r}")
        self.norm = norm

        def make_norm(c):
            if norm == "batch":
                return nn.BatchNorm2d(c)
            if norm == "group":
                return nn.GroupNorm(min(groups, c), c)
            return nn.Identity()

        layers, ch = [], in_channels
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(ch, channels, 3, stride=2, padding=1,
                          padding_mode="circular", bias=(norm == "none")),
                make_norm(channels),
                nn.ReLU(inplace=True),
            ]
            ch = channels
        self.conv = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, mlp_hidden),
            nn.LeakyReLU(0.01, inplace=True) if mlp_act == "leaky" else nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden, n_classes),
        )

    def forward(self, x):
        f = self.conv(x)
        pooled = torch.cat([f.mean(dim=(-2, -1)), f.amax(dim=(-2, -1))], dim=1)
        return self.mlp(pooled)


class MotionDigitClassifier(nn.Module):
    """A backbone's encoder, a pool over the velocity axis, and a conv head."""

    def __init__(self, model="melstm", hidden_channels=32, kernel_size=3,
                 v_range=2, n_slots=2, n_classes=10,
                 velocity_pool="attention", pool_temperature=1.0,
                 velocity_source="bootstrap",
                 head_channels=64, head_blocks=3, head_mlp_hidden=128,
                 head_dropout=0.0, head_norm="batch", input_channels=1,
                 head_mlp_act="relu"):
        super().__init__()
        self.model = model
        self.hidden_channels = hidden_channels
        self.velocity_source = velocity_source
        self.x_track_p = 0.0

        if model in ("lstm", "felstm"):
            if model == "lstm":
                v_range = 0
            self.backbone = Seq2SeqFEConvLSTM(
                input_channels=input_channels, hidden_channels=hidden_channels,
                kernel_size=kernel_size, v_range=v_range, pool_type="max",
                decoder_conv_layers=1, decoder_channels=hidden_channels)
            self.n_velocities = self.backbone.num_v
        elif model == "melstm":
            self.backbone = Seq2SeqMEConvLSTM(
                input_channels=input_channels, hidden_channels=hidden_channels,
                kernel_size=kernel_size, n_slots=n_slots, slot_reduce="max",
                decoder_layers=1, decoder_channels=hidden_channels)
            self.n_velocities = n_slots
        else:
            raise ValueError(f"unknown model {model!r}")

        if velocity_source not in ("tracked", "frame_pair", "bootstrap"):
            raise ValueError(f"unknown velocity_source {velocity_source!r}")
        self._frame_pair_pc = PhaseCorrelation(n_modes=max(1, self.n_velocities))
        self._pc1 = PhaseCorrelation(n_modes=1)

        # pool built last so --velocity_pool cannot change the other modules' initialisation
        head_in = VelocityPool.output_channels(
            velocity_pool, self.n_velocities, hidden_channels)
        self.head = ConvClassifierHead(
            head_in, n_classes=n_classes, channels=head_channels,
            n_blocks=head_blocks, mlp_hidden=head_mlp_hidden, dropout=head_dropout,
            norm=head_norm, mlp_act=head_mlp_act)
        self.pool = VelocityPool(velocity_pool, self.n_velocities,
                                 hidden_channels, temperature=pool_temperature)

    @staticmethod
    def _match_to_slots(cand, v_prev):
        B, K, _ = cand.shape
        cost = (cand[:, :, None, :] - v_prev[:, None, :, :]).abs().sum(-1)
        out = torch.zeros_like(cand)
        taken = torch.zeros(B, K, dtype=torch.bool, device=cand.device)
        rows = torch.arange(B, device=cand.device)
        for k in range(K):
            c = cost[:, :, k].masked_fill(taken, float("inf"))
            j = c.argmin(dim=1)
            out[:, k] = cand[rows, j]
            taken[rows, j] = True
        return out

    @staticmethod
    def _roll_batch(x, v):
        B, C, H, W = x.shape
        dx = v[:, 0].round().long()
        dy = v[:, 1].round().long()
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                                torch.arange(W, device=x.device), indexing="ij")
        sy = (yy.unsqueeze(0) - dy[:, None, None]) % H
        sx = (xx.unsqueeze(0) - dx[:, None, None]) % W
        idx = (sy * W + sx).reshape(B, 1, H * W).expand(-1, C, -1)
        return x.reshape(B, C, H * W).gather(2, idx).view(B, C, H, W)

    def _residual_bootstrap(self, x0, x1, blur=1.5):
        v1 = self._pc1(x0, x1)[0][:, 0]
        R = (x0 - self._roll_batch(x1, -v1)).abs()

        H, W = x0.shape[-2:]
        ky = torch.fft.fftfreq(H, device=x0.device)[:, None]
        kx = torch.fft.fftfreq(W, device=x0.device)[None, :]
        g = torch.exp(-2 * (np.pi * blur) ** 2 * (ky ** 2 + kx ** 2))
        R = torch.fft.ifft2(torch.fft.fft2(R) * g).real
        R = R / (R.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        W1m = self._roll_batch(R, v1)
        a = R * (x0 - x0.mean(dim=(-2, -1), keepdim=True))
        b = W1m * (x1 - x1.mean(dim=(-2, -1), keepdim=True))
        v2 = self._pc1(a, b)[0][:, 0]
        return torch.stack([v1, v2], dim=1)

    def _candidate_velocities(self, x0, x1):
        if self.velocity_source != "bootstrap":
            return self._frame_pair_pc(x0, x1)[0]
        cand = self._residual_bootstrap(x0, x1)
        if self.n_velocities <= 2:
            return cand[:, :self.n_velocities]
        extra = self._frame_pair_pc(x0, x1)[0][:, :self.n_velocities - 2]
        return torch.cat([cand, extra], dim=1)

    def _encode_melstm_frame_pair(self, seq, return_states=False):
        cell = self.backbone.cell
        B, T, C, H, W = seq.shape
        K = self.n_velocities

        h, c = cell.init_hidden(B, K, H, W, seq.device, seq.dtype)
        v = torch.zeros(B, K, 2, device=seq.device, dtype=seq.dtype)
        vels = []
        states = [] if return_states else None

        for t in range(T):
            if t > 0:
                with torch.no_grad():
                    cand = self._candidate_velocities(seq[:, t - 1], seq[:, t])
                cand = cand.to(seq.dtype)
                v = cand if t == 1 else self._match_to_slots(cand, v)
            h, c = cell(seq[:, t], h, c, v)
            if t > 0:
                vels.append(v.detach())
            if return_states:
                states.append(h.mean(dim=2).detach())

        return h, c, vels, states

    def _encode_melstm_mixed(self, seq, p, return_states=False):
        cell = self.backbone.cell
        B, T, C, H, W = seq.shape
        K = self.n_velocities

        h, c = cell.init_hidden(B, K, H, W, seq.device, seq.dtype)
        v = torch.zeros(B, K, 2, device=seq.device, dtype=seq.dtype)
        vels = []
        states = [] if return_states else None

        for t in range(T):
            if t == 1:
                v = self.backbone.bootstrap_velocities(seq[:, 0], seq[:, 1])
            elif t >= 2:
                if p > 0:
                    with torch.no_grad():
                        cand = self._frame_pair_pc(seq[:, t - 1], seq[:, t])[0]
                    v_x = self._match_to_slots(cand.to(seq.dtype), v)
                if p >= 1:
                    v = v_x
                else:
                    v_h = self.backbone.track_velocities(h, seq[:, t])
                    if p <= 0:
                        v = v_h
                    else:
                        use_x = torch.rand(B, device=seq.device) < p
                        v = torch.where(use_x.view(B, 1, 1), v_x, v_h)
            h, c = cell(seq[:, t], h, c, v)
            if t > 0:
                vels.append(v.detach())
            if return_states:
                states.append(h.mean(dim=2).detach())

        return h, c, vels, states

    def encode(self, seq, return_states=False):
        if self.model == "melstm":
            if (self.training and self.x_track_p > 0
                    and self.velocity_source == "tracked"):
                h, _, vels, states = self._encode_melstm_mixed(
                    seq, self.x_track_p, return_states=return_states)
            elif self.velocity_source in ("frame_pair", "bootstrap"):
                h, _, vels, states = self._encode_melstm_frame_pair(
                    seq, return_states=return_states)
            else:
                h, _, _, vels, states = self.backbone.encode(
                    seq, return_states=return_states)
            v = torch.stack(vels, dim=1) if vels else None
        else:
            h, _, states = self.backbone.encode(seq, return_states=return_states)
            v = None
        st = torch.stack(states, dim=1) if states else None
        return (h, v, st) if return_states else (h, v)

    def forward(self, seq, return_aux=False):
        h, velocities = self.encode(seq)[:2]
        features, weights = self.pool(h)
        logits = self.head(features)
        if not return_aux:
            return logits
        return logits, {"velocities": velocities, "pool_weights": weights}

    def backbone_encoder_parameters(self):
        dec = self.backbone.decoder if self.model == "melstm" else self.backbone.decoder_conv
        dec_ids = {id(p) for p in dec.parameters()}
        return [p for p in self.backbone.parameters() if id(p) not in dec_ids]

    def head_parameters(self):
        return list(self.pool.parameters()) + list(self.head.parameters())

    def trainable_parameters(self):
        return self.backbone_encoder_parameters() + self.head_parameters()

    def submodule_report(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        dec = self.backbone.decoder if self.model == "melstm" else self.backbone.decoder_conv
        rows = [
            ("backbone.cell", "recurrent, run on every velocity copy", n(self.backbone.cell)),
            ("pool", f"velocity reduce ({self.pool.mode})", n(self.pool)),
            ("head.conv", f"{len(self.head.conv) // 3} blocks, circular, stride 2", n(self.head.conv)),
            ("head.mlp", "avg+max pool -> logits", n(self.head.mlp)),
        ]
        trained = sum(r[2] for r in rows)
        rows.append(("backbone.decoder", "UNUSED in this task, excluded from the optimizer", n(dec)))
        return rows, trained

    def parameter_report(self):
        n = lambda ps: sum(p.numel() for p in ps)
        enc, hd = n(self.backbone_encoder_parameters()), n(self.head_parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc, "head": hd, "trained": enc + hd,
                "unused_decoder": total - enc - hd,
                "n_velocities": self.n_velocities}

    def describe(self):
        rows, trained = self.submodule_report()
        w = max(len(r[0]) for r in rows)
        lines = [f"architecture : {self.model}  "
                 f"V={self.n_velocities} velocity "
                 f"{'slots' if self.model == 'melstm' else 'copies'}"
                 + (f"  velocity_source={self.velocity_source}"
                    if self.model == "melstm" else "")]
        if self.model == "felstm":
            lines[0] += f"  (lattice v_range covers |v| <= {max(abs(v[0]) for v in self.backbone.cell.v_list)})"
        lines.append("")
        lines.append(f"  {'submodule':<{w}}  {'params':>9}   note")
        lines.append(f"  {'-' * w}  {'-' * 9}   {'-' * 46}")
        for name, note, cnt in rows:
            tag = "" if name != "backbone.decoder" else ""
            lines.append(f"  {name:<{w}}  {cnt:>9,}   {note}{tag}")
        lines.append(f"  {'-' * w}  {'-' * 9}")
        lines.append(f"  {'TRAINED':<{w}}  {trained:>9,}   what the optimizer updates")
        return "\n".join(lines)


@torch.no_grad()
def recompute_bn_stats(model, loader, device, n_batches=50):
    """Re-estimate BatchNorm running statistics on the given batches."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns or n_batches <= 0:
        return 0

    saved = [(bn.momentum, bn.training) for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()

    was_training = model.training
    model.eval()
    for bn in bns:
        bn.train()

    seen = 0
    for batch in loader:
        if seen >= n_batches:
            break
        model(batch[0].to(device))
        seen += 1

    model.train(was_training)
    for bn, (mom, mode) in zip(bns, saved):
        bn.momentum = mom
        bn.train(mode)
    return seen


def build_classifier(cfg):
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    return MotionDigitClassifier(
        model=get("model", "melstm"),
        hidden_channels=get("hidden_size", 32),
        kernel_size=get("kernel_size", 3),
        v_range=get("v_range", 2),
        n_slots=get("num_vel_modes", 2),
        n_classes=get("n_classes", 10),
        velocity_pool=get("velocity_pool", "attention"),
        pool_temperature=get("pool_temperature", 1.0),
        velocity_source=get("velocity_source", "bootstrap"),
        head_channels=get("head_channels", 64),
        head_blocks=get("head_blocks", 3),
        head_mlp_hidden=get("head_mlp_hidden", 128),
        head_dropout=get("head_dropout", 0.0),
        head_norm=get("head_norm", "batch"),
        head_mlp_act=get("head_mlp_act", "relu"),
    )
