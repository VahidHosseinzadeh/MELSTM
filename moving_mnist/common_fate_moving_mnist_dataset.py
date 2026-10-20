import warnings
from contextlib import contextmanager

import numpy as np
import torch

from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset


def band_limited_noise(H, W, corr_len, rng):
    z = rng.standard_normal((H, W))
    if corr_len > 0:
        ky = np.fft.fftfreq(H)[:, None]
        kx = np.fft.fftfreq(W)[None, :]
        k2 = ky ** 2 + kx ** 2
        z = np.real(np.fft.ifft2(np.fft.fft2(z) * np.exp(-2 * (np.pi * corr_len) ** 2 * k2)))
    return (z - z.mean()) / (z.std() + 1e-8)


def glyph_to_mask(glyph, scale=1, thresh=0.3):
    d = np.asarray(glyph)
    d = d.astype(np.float32) if np.issubdtype(d.dtype, np.floating) \
        else d.astype(np.float32) / 255.0
    if scale != 1:
        d = np.kron(d, np.ones((scale, scale), np.float32))
    return (d > thresh).astype(np.float32)


def digit_mask(digit28, H, W, rng, thresh=0.3, scale=1):
    d = glyph_to_mask(digit28, scale=scale, thresh=thresh)
    h, w = d.shape
    if h > H or w > W:
        raise ValueError(
            f"digit is {h}x{w} after scale={scale} but the canvas is {H}x{W}; "
            f"raise image_size or lower digit_scale.")
    m = np.zeros((H, W), np.float32)
    m[:h, :w] = d
    return np.roll(m, (rng.integers(H), rng.integers(W)), axis=(0, 1))


def velocity_schedule(T, vmax, rng, time_varying=False, tau_lo=3, tau_hi=6):
    def draw():
        while True:
            v = rng.integers(-vmax, vmax + 1, size=2)
            if np.any(v):
                return v
    if not time_varying:
        return np.tile(draw(), (T, 1))
    out, t = [], 0
    while t < T:
        v, hold = draw(), rng.integers(tau_lo, tau_hi + 1)
        out += [v] * int(hold)
        t += hold
    return np.array(out[:T])


def cumulative_displacement(v):
    v = np.asarray(v)
    return np.concatenate([np.zeros((1,) + v.shape[1:], v.dtype),
                           np.cumsum(v, axis=0)[:-1]], axis=0)


def render_sequence(masks, textures, bg_texture, d_fig, d_bg, variant):
    masks = np.asarray(masks, np.float32)
    textures = np.asarray(textures, np.float32)
    N, H, W = masks.shape
    T = len(d_bg)
    if variant not in ("moving_mask", "static_mask"):
        raise ValueError(variant)

    frames = np.empty((T, H, W), np.float32)
    track = np.empty((T, N, H, W), np.float32)
    for t in range(T):
        frame = np.roll(bg_texture, tuple(d_bg[t]), (0, 1))
        for i in range(N):
            tex = np.roll(textures[i], tuple(d_fig[t, i]), (0, 1))
            mt = (np.roll(masks[i], tuple(d_fig[t, i]), (0, 1))
                  if variant == "moving_mask" else masks[i])
            frame = np.where(mt > 0.5, tex, frame)
            track[t, i] = mt
        frames[t] = frame
    return frames, track


def make_sequence(digit28, T=20, H=64, W=64, vmax=3, corr_len=1.0, scale=1,
                  variant="moving_mask", time_varying=False, min_dv=2,
                  thresh=0.3, max_vel_tries=200, rng=None):
    rng = rng or np.random.default_rng()
    m = digit_mask(digit28, H, W, rng, thresh=thresh, scale=scale)
    A = band_limited_noise(H, W, corr_len, rng)
    B = band_limited_noise(H, W, corr_len, rng)

    separated = False
    for _ in range(max_vel_tries):
        v_fg = velocity_schedule(T, vmax, rng, time_varying)
        v_bg = velocity_schedule(T, vmax, rng, time_varying)
        if np.min(np.abs(v_fg - v_bg).max(axis=1)) >= min_dv:
            separated = True
            break

    d_fg = cumulative_displacement(v_fg)
    d_bg = cumulative_displacement(v_bg)

    frames, track = render_sequence(m[None], A[None], B,
                                    d_fig=d_fg[:, None], d_bg=d_bg,
                                    variant=variant)
    return dict(frames=frames, mask=m, mask_track=track[:, 0],
                v_fg=v_fg, v_bg=v_bg, d_fg=d_fg, d_bg=d_bg,
                velocity_separated=separated)


def _yx_to_xy(v):
    return np.ascontiguousarray(np.asarray(v)[..., ::-1])


def _xy_to_yx(v):
    return np.ascontiguousarray(np.asarray(v)[..., ::-1])


class _ParentRNG:
    def __init__(self, ds):
        self._ds = ds

    def _src(self):
        return np.random if self._ds.random else self._ds.rng

    def standard_normal(self, size=None):
        return self._src().standard_normal(size)

    def integers(self, low, high=None, size=None):
        lo, hi = (0, low) if high is None else (low, high)
        return self._src().randint(lo, hi, size=size)


class CommonFateMovingMNISTDataset(TDMovingMNISTDataset):
    """Common-fate Moving MNIST: a digit defined only by its motion against a moving background."""

    def __init__(
        self,
        root,
        train=True,
        seq_len=20,
        image_size=64,

        num_figures=1,
        variant="moving_mask",
        corr_len=0.0,
        digit_scale=1,
        mask_threshold=0.3,
        min_dv=2,
        separate_figures=False,
        bg_opposite_at_start=False,
        bg_speed_range=None,
        max_velocity_tries=200,
        normalize="affine",
        clip=3.0,

        max_speed=3,
        motion_mode=None,
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

        digit_indices=None,

        return_motion=True,
        return_positions=False,
        return_mask=False,

        transform=None,
        download=True,
        random=True,
        seed=42,
        max_tries=200,
    ):
        if variant not in ("moving_mask", "static_mask"):
            raise ValueError(
                f"variant must be 'moving_mask' or 'static_mask', got {variant!r}")
        if normalize not in ("affine", "minmax", "none"):
            raise ValueError(
                f"normalize must be 'affine', 'minmax' or 'none', got {normalize!r}")
        if num_figures < 1:
            raise ValueError(f"num_figures must be >= 1, got {num_figures}")
        if min_dv > 2 * max_speed and bg_speed_range is None:
            raise ValueError(
                f"min_dv={min_dv} is unsatisfiable with max_speed={max_speed} "
                f"(the largest possible separation is {2 * max_speed}).")

        if bg_speed_range is not None:
            lo, hi = bg_speed_range
            if lo > hi:
                raise ValueError(f"bg_speed_range={bg_speed_range} is empty (lo > hi).")
            if lo <= max_speed:
                raise ValueError(
                    f"bg_speed_range={bg_speed_range} overlaps the figure grid "
                    f"(max_speed={max_speed}); the whole point of this argument is "
                    f"that the background can NEVER coincide with a figure, so its "
                    f"minimum speed must exceed max_speed.")

        if motion_mode is None:
            motion_mode = "piecewise" if motion_difficulty is not None else "constant"

        super().__init__(
            root=root,
            train=train,
            seq_len=seq_len,
            image_size=image_size,
            num_digits=num_figures + 1,
            max_speed=max_speed,
            motion_mode=motion_mode,
            transition_mode=transition_mode,
            min_segment=min_segment,
            max_segment=max_segment,
            p_change=p_change,
            smooth_probability=smooth_probability,
            neighbor_kernel=neighbor_kernel,
            motion_difficulty=motion_difficulty,
            freeze_after=freeze_after,
            min_center_distance=min_center_distance,
            reject_overlap=reject_overlap,
            require_distinct_velocities=require_distinct_velocities,
            require_distinct_digits=require_distinct_digits,
            return_motion=return_motion,
            return_positions=return_positions,
            transform=transform,
            download=download,
            random=random,
            seed=seed,
            max_tries=max_tries,
        )

        self.num_figures = num_figures
        self.digit_indices = (None if digit_indices is None
                              else list(int(i) for i in digit_indices))
        self.variant = variant
        self.corr_len = corr_len
        self.digit_scale = digit_scale
        self.mask_threshold = mask_threshold
        self.min_dv = min_dv
        self.separate_figures = separate_figures
        self.bg_opposite_at_start = bg_opposite_at_start
        self.bg_speed_range = bg_speed_range

        self.bg_velocity_grid = None
        self.bg_separation_is_structural = False
        if bg_speed_range is not None:
            lo, hi = bg_speed_range
            self.bg_velocity_grid = [
                (vx, vy)
                for vx in range(-hi, hi + 1)
                for vy in range(-hi, hi + 1)
                if lo <= max(abs(vx), abs(vy)) <= hi
            ]
            self.bg_separation_is_structural = (lo - max_speed) >= min_dv
        self.max_velocity_tries = max_velocity_tries
        self.normalize = normalize
        self.clip = float(clip)
        self.return_mask = return_mask

        self._rng = _ParentRNG(self)
        self._warned_unseparated = False

    def __len__(self):
        return len(self.mnist) if self.digit_indices is None else len(self.digit_indices)

    def __getitem__(self, index):
        S, N = self.image_size, self.num_figures

        glyphs, labels = self._sample_glyphs(index)
        positions = self._sample_initial_positions(N)
        motions, separated = self._sample_separated_motion()

        if not separated and not self._warned_unseparated:
            self._warned_unseparated = True
            warnings.warn(
                f"could not separate the layer velocities by min_dv={self.min_dv} "
                f"within {self.max_velocity_tries} tries (motion_mode="
                f"{self.motion_mode!r}, max_speed={self.max_speed}, "
                f"num_figures={N}, separate_figures={self.separate_figures}); "
                f"such sequences contain little or no motion-defined figure. "
                f"Lower min_dv, raise max_speed, or raise max_velocity_tries.",
                UserWarning, stacklevel=2)

        disp = cumulative_displacement(motions.numpy())

        masks = np.stack([self._figure_mask(g, cx, cy)
                          for g, (cx, cy) in zip(glyphs, positions)])
        textures = np.stack([band_limited_noise(S, S, self.corr_len, self._rng)
                             for _ in range(N)])
        bg_texture = band_limited_noise(S, S, self.corr_len, self._rng)

        frames, track = render_sequence(
            masks, textures, bg_texture,
            d_fig=_xy_to_yx(disp[:, :N]),
            d_bg=_xy_to_yx(disp[:, N]),
            variant=self.variant)

        seq = torch.from_numpy(self._normalize(frames)).unsqueeze(1)
        if self.transform:
            seq = self.transform(seq)

        out = [seq, (labels[0] if N == 1 else torch.tensor(labels, dtype=torch.long))]
        if self.return_motion:
            out.append(motions)
        if self.return_positions:
            pos = torch.from_numpy(disp[:, :N]).long() + torch.tensor(
                positions, dtype=torch.long).unsqueeze(0)
            out.append(pos % S)
        if self.return_mask:
            out.append(torch.from_numpy(track))
        return tuple(out)

    def _sample_glyphs(self, index):
        pool = self.digit_indices
        glyphs, labels, used = [], [], set()
        for i in range(self.num_figures):
            if i == 0 and pool is not None:
                mnist_idx = pool[index % len(pool)]
                img, lbl = self.mnist[mnist_idx]
                if self.require_distinct_digits:
                    used.add(lbl)
                glyphs.append(np.asarray(img, dtype=np.uint8))
                labels.append(lbl)
                continue
            while True:
                if pool is None:
                    mnist_idx = self._randint(0, len(self.mnist))
                else:
                    mnist_idx = pool[self._randint(0, len(pool))]
                img, lbl = self.mnist[mnist_idx]
                if (not self.require_distinct_digits) or (lbl not in used):
                    break
            glyphs.append(np.asarray(img, dtype=np.uint8))
            labels.append(lbl)
            used.add(lbl)
        return glyphs, labels

    def _figure_mask(self, glyph, cx, cy):
        S = self.image_size
        g = glyph_to_mask(glyph, scale=self.digit_scale, thresh=self.mask_threshold)
        h, w = g.shape
        if h > S or w > S:
            raise ValueError(
                f"digit is {h}x{w} after digit_scale={self.digit_scale} but the "
                f"canvas is {S}x{S}; raise image_size or lower digit_scale.")
        m = np.zeros((S, S), np.float32)
        top, left = (S - h) // 2, (S - w) // 2
        m[top:top + h, left:left + w] = g
        return np.roll(m, (cy, cx), (0, 1))

    def _velocities_separated(self, motions):
        if self.min_dv <= 0:
            return True
        N = self.num_figures
        fig, bg = motions[:, :N], motions[:, N:]
        if not self.bg_separation_is_structural:
            if int((fig - bg).abs().amax(dim=2).min()) < self.min_dv:
                return False
        if self.separate_figures and N > 1:
            for i in range(N):
                for j in range(i + 1, N):
                    gap = (fig[:, i] - fig[:, j]).abs().amax(dim=1).min()
                    if int(gap) < self.min_dv:
                        return False

        if self.bg_opposite_at_start:
            # t = 0: the background moves opposite to every figure (negative dot product)
            f0, b0 = motions[0, :N].to(torch.long), motions[0, N].to(torch.long)
            if bool(((f0 * b0.unsqueeze(0)).sum(dim=1) >= 0).any()):
                return False

        return True

    @contextmanager
    def _velocity_grid(self, grid, n_slots):
        saved = (self.velocity_grid, self._velocity_set, self.num_digits)
        self.velocity_grid, self._velocity_set, self.num_digits = (
            grid, set(grid), n_slots)
        try:
            yield
        finally:
            self.velocity_grid, self._velocity_set, self.num_digits = saved

    def _generate_split_grid_motion(self):
        N = self.num_figures
        with self._velocity_grid(self.velocity_grid, N):
            fig = self._generate_motion_trajectory()
        with self._velocity_grid(self.bg_velocity_grid, 1):
            bg = self._generate_motion_trajectory()
        return torch.cat([fig, bg], dim=1)

    def _sample_separated_motion(self):
        draw = (self._generate_split_grid_motion if self.bg_velocity_grid is not None
                else self._generate_motion_trajectory)
        motions = None
        for _ in range(self.max_velocity_tries):
            motions = draw()
            if self._velocities_separated(motions):
                return motions, True
        return motions, False

    def _normalize(self, frames):
        if self.normalize == "none":
            return frames.astype(np.float32)
        if self.normalize == "affine":
            c = self.clip
            return (np.clip(frames, -c, c) / (2 * c) + 0.5).astype(np.float32)
        lo, hi = float(frames.min()), float(frames.max())
        return ((frames - lo) / (hi - lo + 1e-8)).astype(np.float32)
