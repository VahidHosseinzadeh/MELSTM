import math

import numpy as np

from motion_benchmarks import _repo
from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset

GENERATOR_MODES = ("piecewise", "stochastic", "accelerate")
MODES = ("none", "constant") + GENERATOR_MODES + ("shake",)


class _VelocityProcess(TDMovingMNISTDataset):
    def __init__(self, seq_len, max_speed, motion_mode, transition_mode, min_segment,
                 max_segment, smooth_probability=0.8, p_change=0.25, neighbor_kernel="legacy",
                 include_zero=True):
        self.seq_len = seq_len
        self.num_digits = 1
        self.max_speed = max_speed
        self.motion_mode = motion_mode
        self.transition_mode = transition_mode
        self.min_segment = min_segment
        self.max_segment = max_segment
        self.smooth_probability = smooth_probability
        self.p_change = p_change
        self.freeze_after = None
        self.require_distinct_velocities = False
        self.random = False
        self.rng = np.random.RandomState(0)
        self.velocity_grid = [(vx, vy)
                              for vx in range(-max_speed, max_speed + 1)
                              for vy in range(-max_speed, max_speed + 1)
                              if include_zero or (vx, vy) != (0, 0)]
        self.neighbor_kernel = neighbor_kernel
        self._velocity_set = set(self.velocity_grid)
        self._unit_offsets = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                              if (a, b) != (0, 0)]

    def draw(self, rng):
        self.rng = rng
        return self._generate_motion_trajectory()[:, 0].numpy().astype(np.int64)


class CameraMotion:
    """A camera-motion law: draw(rng) returns one clip's (seq_len, 2) velocity sequence."""

    def __init__(self, mode="none", v_range=1, seq_len=16, transition="uniform",
                 min_segment=3, max_segment=6, neighbor_kernel="legacy",
                 smooth_probability=0.8, p_change=0.25,
                 shake_amp=(1.0, 3.0), shake_period=(6.0, 16.0), shake_axes="both",
                 shake_vmax=1, keller_draws=True, include_zero=True):
        if mode not in MODES:
            raise ValueError(f"unknown camera mode {mode!r}; expected one of {MODES}")
        if shake_axes not in ("both", "x", "y"):
            raise ValueError(f"shake_axes {shake_axes!r}: expected both, x or y")
        self.mode = mode
        self.v_range = int(v_range)
        self.seq_len = int(seq_len)
        self.keller_draws = bool(keller_draws)
        self.shake_amp = (float(shake_amp[0]), float(shake_amp[1]))
        self.shake_period = (float(shake_period[0]), float(shake_period[1]))
        self.shake_axes = shake_axes
        self.shake_vmax = shake_vmax
        self.include_zero = bool(include_zero)
        self._process = None
        if mode in GENERATOR_MODES or (mode == "constant" and not self.include_zero):
            self._process = _VelocityProcess(
                seq_len, self.v_range, mode, transition, min_segment, max_segment,
                smooth_probability=smooth_probability, p_change=p_change,
                neighbor_kernel=neighbor_kernel, include_zero=self.include_zero)
        self.transition = transition
        self.min_segment, self.max_segment = min_segment, max_segment

    @property
    def uses_keller_draws(self):
        return self.mode == "constant" and self.keller_draws and self.include_zero

    def draw(self, rng):
        T = self.seq_len
        if self.mode == "none":
            return np.zeros((T, 2), dtype=np.int64)
        if self._process is not None:
            return self._process.draw(rng)
        if self.mode == "constant":
            v = (rng.randint(-self.v_range, self.v_range + 1),
                 rng.randint(-self.v_range, self.v_range + 1))
            return np.tile(np.asarray(v, dtype=np.int64), (T, 1))
        if self.mode == "shake":
            v = np.zeros((T, 2), dtype=np.int64)
            for axis in (0, 1):
                if self.shake_axes == "both" or self.shake_axes == "xy"[axis]:
                    v[:, axis] = self._shake_axis(rng, T)
            return v
        raise AssertionError(f"no draw rule for mode {self.mode!r}")

    def _shake_axis(self, rng, T):
        a_lo, a_hi = self.shake_amp
        for _ in range(100):
            P = rng.uniform(*self.shake_period)
            cap = a_hi
            if self.shake_vmax:
                cap = min(a_hi, self.shake_vmax / (2.0 * math.sin(math.pi / P)))
            if cap >= a_lo:
                break
        A = rng.uniform(a_lo, cap) if cap > a_lo else cap
        phi = rng.uniform(0.0, 2.0 * math.pi)
        t = np.arange(T + 1)
        d = np.floor(A * np.sin(2.0 * math.pi * t / P + phi) + 0.5)
        return np.diff(d).astype(np.int64)

    def describe(self):
        if self.mode == "none":
            return "none (static camera)"
        zero = "" if self.include_zero else " without (0,0)"
        if self.mode == "constant":
            src = "Keller's draws" if self.uses_keller_draws else "seeded draws"
            return f"constant, one v per clip from V_{self.v_range}{zero} ({src})"
        if self.mode == "shake":
            return (f"shake, A in [{self.shake_amp[0]:g}, {self.shake_amp[1]:g}] px, "
                    f"P in [{self.shake_period[0]:g}, {self.shake_period[1]:g}] frames, "
                    f"axes {self.shake_axes}, |v| <= {self.shake_vmax or 'inf'}")
        zero = "incl. (0,0)" if self.include_zero else "without (0,0)"
        return (f"{self.mode} on V_{self.v_range} {zero}, transition {self.transition}, "
                f"segments {self.min_segment}-{self.max_segment}")


def keller_constant_velocities(n_clips, v_range):
    rng = np.random.RandomState(42)
    out = np.zeros((n_clips, 2), dtype=np.int64)
    for i in range(n_clips):
        vx = rng.randint(-v_range, v_range + 1)
        vy = rng.randint(-v_range, v_range + 1)
        out[i] = (-vx, -vy)
    return out


def displacements(v):
    d = np.zeros_like(v)
    d[1:] = np.cumsum(v[:-1], axis=0)
    return d
