import warnings

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class TDMovingMNISTDataset(Dataset):
    """Moving MNIST with time-dependent, multi-digit motion."""

    def __init__(
        self,
        root,
        train=True,
        seq_len=20,
        image_size=64,
        num_digits=2,

        max_speed=3,

        motion_mode="piecewise",
        transition_mode="smooth",

        min_segment=3,
        max_segment=6,

        p_change=0.25,
        smooth_probability=0.8,
        neighbor_kernel="legacy",

        motion_difficulty=None,

        freeze_after=None,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,
        require_distinct_digits=True,

        return_motion=True,
        return_positions=False,

        transform=None,
        download=True,

        random=True,
        seed=42,

        max_tries=200,
    ):
        super().__init__()

        self.mnist     = MNIST(root=root, train=train, download=download)
        self.to_tensor = ToTensor()

        self.seq_len    = seq_len
        self.image_size = image_size
        self.num_digits = num_digits
        self.max_speed  = max_speed

        self.motion_mode        = motion_mode
        self.transition_mode    = transition_mode
        self.min_segment        = min_segment
        self.max_segment        = max_segment
        self.p_change           = p_change
        self.smooth_probability = smooth_probability
        self.freeze_after       = freeze_after

        self.min_center_distance         = min_center_distance
        self.reject_overlap              = reject_overlap
        self.require_distinct_velocities = require_distinct_velocities
        self.require_distinct_digits     = require_distinct_digits

        self.return_motion    = return_motion
        self.return_positions = return_positions

        self.transform = transform
        self.random    = random
        self.max_tries = max_tries

        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.velocity_grid = [
            (vx, vy)
            for vx in range(-max_speed, max_speed + 1)
            for vy in range(-max_speed, max_speed + 1)
            if (vx, vy) != (0, 0)
        ]

        self.neighbor_kernel = neighbor_kernel
        self._velocity_set   = set(self.velocity_grid)
        self._unit_offsets   = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                                if (a, b) != (0, 0)]

        if motion_difficulty is not None:
            overridden = []
            if motion_mode != "piecewise":
                overridden.append(f"motion_mode={motion_mode!r}")
            if transition_mode != "smooth":
                overridden.append(f"transition_mode={transition_mode!r}")
            if overridden:
                warnings.warn(
                    "motion_difficulty is defined on the stochastic family and "
                    "overrides the motion family, so "
                    + " and ".join(overridden)
                    + " will be ignored in favour of motion_mode='stochastic', "
                      "transition_mode='smooth'.",
                    UserWarning, stacklevel=2)
            self._apply_difficulty(motion_difficulty)

    def _apply_difficulty(self, d):
        d = float(np.clip(d, 0.0, 1.0))

        self.motion_mode        = "stochastic"
        self.transition_mode    = "smooth"
        self.p_change           = d ** 2.170
        self.smooth_probability = 0.80

        mean_gap = (1.0 / self.p_change) if self.p_change > 1e-9 else float(self.seq_len)
        mean_gap = min(mean_gap, float(self.seq_len))
        self.min_segment = max(1, int(round(0.70 * mean_gap)))
        self.max_segment = max(self.min_segment + 1, int(round(1.35 * mean_gap)))

    def reset_rng(self):
        self.rng = np.random.RandomState(self.seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        S = self.image_size
        N = self.num_digits
        T = self.seq_len

        imgs, labels = [], []
        used_labels = set()

        for _ in range(N):
            while True:
                idx = self._randint(0, len(self.mnist))
                img_pil, lbl = self.mnist[idx]

                if (not self.require_distinct_digits) or (lbl not in used_labels):
                    break

            imgs.append(self._pad_digit(self.to_tensor(img_pil), S))
            labels.append(lbl)
            used_labels.add(lbl)

        init_pos = self._sample_initial_positions()
        motions  = self._generate_motion_trajectory()

        cum = torch.cumsum(motions, dim=0)
        cum = torch.cat([torch.zeros(1, N, 2, dtype=torch.long), cum[:-1]], dim=0)

        init_tensor = torch.tensor(init_pos, dtype=torch.long)
        positions   = cum + init_tensor.unsqueeze(0)

        seq = torch.zeros(T, 1, S, S, dtype=imgs[0].dtype)
        for t in range(T):
            frame = torch.zeros(1, S, S)
            for i, img in enumerate(imgs):
                cx = int(positions[t, i, 0].item())
                cy = int(positions[t, i, 1].item())
                frame += torch.roll(img, shifts=(cy, cx), dims=(1, 2))
            seq[t] = frame.clamp(0.0, 1.0)

        if self.transform:
            seq = self.transform(seq)

        lbl_out = (labels[0] if N == 1
                   else torch.tensor(labels, dtype=torch.long))
        out = [seq, lbl_out]
        if self.return_motion:
            out.append(motions)
        if self.return_positions:
            out.append(positions % S)
        return tuple(out)

    def _pad_digit(self, img, S):
        _, h, w    = img.shape
        pad_top    = (S - h) // 2
        pad_bottom = S - h - pad_top
        pad_left   = (S - w) // 2
        pad_right  = S - w - pad_left
        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))

    def _toroidal_dist(self, ax, ay, bx, by):
        S  = self.image_size
        dx = min(abs(ax - bx), S - abs(ax - bx))
        dy = min(abs(ay - by), S - abs(ay - by))
        return np.sqrt(dx * dx + dy * dy)

    def _position_is_valid(self, cx, cy, existing):
        digit_sz  = 28
        threshold = float(self.min_center_distance)
        if self.reject_overlap:
            threshold = max(threshold, float(digit_sz))

        for pcx, pcy in existing:
            if self._toroidal_dist(cx, cy, pcx, pcy) < threshold:
                return False
        return True

    def _sample_initial_positions(self, n=None):
        S = self.image_size
        N = self.num_digits if n is None else n
        positions = []

        for i in range(N):
            placed = False
            for _ in range(self.max_tries):
                cx = self._randint(0, S)
                cy = self._randint(0, S)
                if self._position_is_valid(cx, cy, positions):
                    positions.append((cx, cy))
                    placed = True
                    break

            if not placed:
                if positions:
                    ax, ay = positions[0]
                else:
                    ax = self._randint(0, S)
                    ay = self._randint(0, S)

                cx = (ax + i * S // N) % S
                cy = (ay + i * S // N) % S
                positions.append((cx, cy))

        return positions

    def _generate_motion_trajectory(self):
        T = self.seq_len
        N = self.num_digits

        motions = torch.zeros(T, N, 2, dtype=torch.long)

        forbidden = []
        for i in range(N):
            v = self._sample_velocity(
                forbidden=forbidden if self.require_distinct_velocities else []
            )
            motions[0, i, 0] = v[0]
            motions[0, i, 1] = v[1]
            forbidden.append(v)

        if self.motion_mode == "constant":
            motions[1:] = motions[0]
            return self._apply_freeze(motions)

        segment_remaining = [
            self._randint(self.min_segment, self.max_segment + 1)
            for _ in range(N)
        ]
        accel_sign = ([self._choice([+1, +1, -1]) for _ in range(N)]
                      if self.motion_mode == "accelerate" else None)

        for t in range(1, T):
            for i in range(N):
                current = (int(motions[t-1, i, 0]), int(motions[t-1, i, 1]))
                new_v   = current

                if self.motion_mode == "piecewise":
                    segment_remaining[i] -= 1
                    if segment_remaining[i] <= 0:
                        new_v = self._update_velocity(current)
                        segment_remaining[i] = self._randint(
                            self.min_segment, self.max_segment + 1
                        )
                elif self.motion_mode == "stochastic":
                    if self._random() < self.p_change:
                        new_v = self._update_velocity(current)
                elif self.motion_mode == "accelerate":
                    segment_remaining[i] -= 1
                    if segment_remaining[i] <= 0:
                        s = accel_sign[i]
                        step = lambda c: c + s * (1 if c >= 0 else -1)
                        cand = (min(max(step(current[0]), -self.max_speed), self.max_speed),
                                min(max(step(current[1]), -self.max_speed), self.max_speed))
                        new_v = current if cand == (0, 0) else cand
                        speed = max(abs(new_v[0]), abs(new_v[1]))
                        if speed >= self.max_speed:
                            accel_sign[i] = -1
                        elif speed <= 1:
                            accel_sign[i] = +1
                        segment_remaining[i] = self._randint(
                            self.min_segment, self.max_segment + 1
                        )
                else:
                    raise ValueError(
                        f"Unknown motion_mode '{self.motion_mode}'. Choose from "
                        f"'constant', 'piecewise', 'stochastic', 'accelerate'."
                    )

                motions[t, i, 0] = new_v[0]
                motions[t, i, 1] = new_v[1]

        return self._apply_freeze(motions)

    def _apply_freeze(self, motions):
        if self.freeze_after is None:
            return motions

        T = motions.shape[0]
        f = int(np.clip(self.freeze_after, 1, T))

        if f >= 2:
            last_v = motions[f - 2].clone()
            motions[f - 1:] = last_v.unsqueeze(0).expand(T - (f - 1), -1, -1)

        return motions

    def _sample_velocity(self, forbidden=None):
        forbidden  = forbidden or []
        candidates = [v for v in self.velocity_grid if v not in forbidden]
        if not candidates:
            return self._choice(self.velocity_grid)
        return self._choice(candidates)

    def _sample_neighbor_velocity(self, velocity):
        if self.neighbor_kernel == "symmetric":
            dx, dy = self._choice(self._unit_offsets)
            cand = (velocity[0] + dx, velocity[1] + dy)
            return cand if cand in self._velocity_set else velocity

        vx, vy = velocity
        neighbors = [
            u for u in self.velocity_grid
            if u != velocity
            and abs(u[0] - vx) <= 1
            and abs(u[1] - vy) <= 1
        ]
        if not neighbors:
            return self._sample_velocity(forbidden=[velocity])
        return self._choice(neighbors)

    def _update_velocity(self, velocity):
        if self.transition_mode == "uniform":
            return self._sample_velocity(forbidden=[velocity])
        if self._random() < self.smooth_probability:
            return self._sample_neighbor_velocity(velocity)
        return self._sample_velocity(forbidden=[velocity])

    def _randint(self, low, high):
        if self.random:
            return int(np.random.randint(low, high))
        return int(self.rng.randint(low, high))

    def _choice(self, values):
        idx = (np.random.randint(len(values)) if self.random
               else self.rng.randint(len(values)))
        return values[idx]

    def _random(self):
        return np.random.rand() if self.random else self.rng.rand()
