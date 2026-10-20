import torch

from motion_benchmarks import _repo
from motion_benchmarks.models.cf_classifier import MotionVideoClassifier
from motion_classification_model import MotionDigitClassifier

from .kth_dataset import KTH_ACTIONS


class KTHClassifier(MotionVideoClassifier):
    def __init__(self, *args, slot_assign="nearest", static_slot=False, **kwargs):
        super().__init__(*args, **kwargs)
        if slot_assign not in ("nearest", "shift", "anchored"):
            raise ValueError(f"slot_assign {slot_assign!r}: expected nearest, shift or anchored")
        if static_slot and (self.model != "melstm" or self.n_velocities < 2):
            raise ValueError("static_slot needs melstm with >= 2 slots")
        self.slot_assign = slot_assign
        self.static_slot = bool(static_slot)

    def forward(self, seq, return_aux=False):
        if self.readout_steps <= 1:
            return MotionDigitClassifier.forward(self, seq, return_aux=return_aux)
        states = []
        handle = self.backbone.cell.register_forward_hook(
            lambda _module, _inputs, out: states.append(out[0]))
        try:
            velocities = self.encode(seq)[1]
        finally:
            handle.remove()
        logits, weights = [], None
        for h in states[-self.readout_steps:]:
            features, weights = self.pool(h)
            logits.append(self.head(features))
        logits = torch.stack(logits, dim=0).mean(dim=0)
        if not return_aux:
            return logits
        return logits, {"velocities": velocities, "pool_weights": weights}

    def encode(self, seq, return_states=False):
        if self.model == "melstm" and self.static_slot and self.velocity_source == "tracked":
            p = self.x_track_p if self.training else 0.0
            h, vels, states = self._encode_static_tracked(seq, p, return_states)
            v = torch.stack(vels, dim=1)
            st = torch.stack(states, dim=1) if states else None
            return (h, v, st) if return_states else (h, v)
        return super().encode(seq, return_states=return_states)

    def _encode_static_tracked(self, seq, p, return_states=False):
        cell = self.backbone.cell
        B, T, C, H, W = seq.shape
        K = self.n_velocities
        h, c = cell.init_hidden(B, K, H, W, seq.device, seq.dtype)
        rest = torch.zeros(B, 1, 2, device=seq.device, dtype=seq.dtype)
        v = torch.zeros(B, K, 2, device=seq.device, dtype=seq.dtype)
        vels, states = [], ([] if return_states else None)
        for t in range(T):
            if t == 1:
                boot = self.backbone.bootstrap_velocities(seq[:, 0], seq[:, 1]).to(seq.dtype)
                v = torch.cat([rest, boot[:, :K - 1]], dim=1)
            elif t >= 2:
                if p > 0:
                    with torch.no_grad():
                        cand = self._frame_pair_pc(seq[:, t - 1], seq[:, t])[0]
                    v_x = self._assign(cand[:, :K - 1].to(seq.dtype), v[:, 1:])
                if p >= 1:
                    v_new = v_x
                else:
                    v_h = self.backbone.track_velocities(h[:, 1:], seq[:, t]).to(seq.dtype)
                    if p <= 0:
                        v_new = v_h
                    else:
                        use_x = torch.rand(B, device=seq.device) < p
                        v_new = torch.where(use_x.view(B, 1, 1), v_x, v_h)
                v = torch.cat([rest, v_new], dim=1)
            h, c = cell(seq[:, t], h, c, v)
            if t > 0:
                vels.append(v.detach())
            if return_states:
                states.append(h.mean(dim=2).detach())
        return h, vels, states

    def _candidate_velocities(self, x0, x1):
        cand = super()._candidate_velocities(x0, x1)
        if self.static_slot:
            cand = torch.cat([torch.zeros_like(cand[:, :1]), cand[:, :-1]], dim=1)
        return cand

    def _assign(self, cand, v_prev):
        match = MotionDigitClassifier._match_to_slots
        if self.slot_assign == "anchored" and cand.shape[1] > 1:
            return torch.cat([cand[:, :1], match(cand[:, 1:], v_prev[:, 1:])], dim=1)
        if self.slot_assign == "shift":
            return match_up_to_shift(cand, v_prev)
        return match(cand, v_prev)

    def _match_to_slots(self, cand, v_prev):
        if self.static_slot:
            return torch.cat([cand[:, :1], self._assign(cand[:, 1:], v_prev[:, 1:])], dim=1)
        return self._assign(cand, v_prev)

    def copy_labels(self):
        if self.model == "felstm":
            return [f"({vx},{vy})" for vx, vy in self.backbone.cell.v_list]
        if self.model == "melstm":
            return [("s0 (0,0)" if self.static_slot and k == 0 else f"s{k}")
                    for k in range(self.n_velocities)]
        return ["h"]

    def pool_label(self):
        label = {"max": "max over V", "attention": "attention\npool", "mean": "mean over V",
                 "concat": "concat V\n(ch. mean)"}[self.pool.mode]
        if self.readout_steps > 1:
            return label + f"\n(head input,\nlast {self.readout_steps} steps)"
        return label + "\n(head input)"


def match_up_to_shift(cand, v_prev):
    B, K, _ = cand.shape
    deltas = (cand[:, :, None, :] - v_prev[:, None, :, :]).reshape(B, K * K, 2)
    best = cand.clone()
    best_cost = torch.full((B,), float("inf"), dtype=cand.dtype, device=cand.device)
    for i in range(K * K):
        target = v_prev + deltas[:, i:i + 1, :]
        out = MotionDigitClassifier._match_to_slots(cand, target)
        cost = (out - target).abs().sum(dim=(-2, -1))
        better = cost < best_cost
        best = torch.where(better[:, None, None], out, best)
        best_cost = torch.where(better, cost, best_cost)
    return best


def phase_corr_kwargs(cfg):
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    pc = {}
    if get("pc_subpixel", 0):
        pc["subpixel"] = True
    if get("pc_suppress_radius", 0):
        pc["suppress_radius"] = int(get("pc_suppress_radius"))
    r = get("pc_search_radius", None)
    if r is not None and int(r) > 0:
        pc["search_radius"] = int(r)
    if get("pc_alpha", 1.0) != 1.0:
        pc["alpha"] = float(get("pc_alpha"))
    return pc


def build_kth_classifier(cfg):
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    pc = phase_corr_kwargs(cfg)
    net = KTHClassifier(
        model=get("model", "melstm"),
        hidden_channels=get("hidden_size", 64),
        kernel_size=get("kernel_size", 3),
        v_range=get("v_range", 1),
        n_slots=get("num_vel_modes", 4),
        n_classes=len(KTH_ACTIONS),
        velocity_pool=get("velocity_pool", "max"),
        velocity_source=get("velocity_source", "frame_pair"),
        head_channels=get("head_channels", 64),
        head_blocks=get("head_blocks", 3),
        head_mlp_hidden=get("head_mlp_hidden", 128),
        head_dropout=get("head_dropout", 0.0),
        head_norm=get("head_norm", "batch"),
        head_mlp_act=get("head_mlp_act", "leaky"),
        phase_corr_kwargs=pc,
        forget_bias=get("forget_bias", None),
        slot_assign=get("slot_assign", "nearest"),
        static_slot=bool(get("static_slot", 0)),
        readout_steps=int(get("readout_steps", 1)),
    )
    if net.model == "melstm":
        net.backbone.cell.integer_shift = not pc.get("subpixel", False)
    return net


@torch.no_grad()
def record_states(model, seq):
    rec = []

    def hook(_module, _inputs, out):
        h = out[0]
        B, V, C, H, W = h.shape
        feat, w = model.pool(h)
        hmax, arg = h.max(dim=1)
        counts = torch.zeros(B, V, H, W, device=h.device).scatter_add_(
            1, arg, torch.ones_like(hmax))
        rec.append((h.mean(dim=2), feat.mean(dim=1), counts.argmax(dim=1),
                    counts.sum(dim=(-2, -1)) / (C * H * W), w))

    was_training = model.training
    model.eval()
    handle = model.backbone.cell.register_forward_hook(hook)
    try:
        velocities = model.encode(seq)[1]
    finally:
        handle.remove()
    logits = model(seq)
    model.train(was_training)
    return {
        "copies": torch.stack([r[0] for r in rec], dim=1).cpu(),
        "pooled": torch.stack([r[1] for r in rec], dim=1).cpu(),
        "winner": torch.stack([r[2] for r in rec], dim=1).cpu(),
        "share": rec[-1][3].cpu(),
        "weights": (None if rec[-1][4] is None
                    else torch.stack([r[4] for r in rec], dim=1).cpu()),
        "velocities": None if velocities is None else velocities.cpu(),
        "logits": logits.cpu(),
    }
