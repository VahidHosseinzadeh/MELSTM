import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .camera_motion import CameraMotion, displacements, keller_constant_velocities

KTH_ACTIONS = ("walking", "jogging", "running", "boxing", "handwaving", "handclapping")
SPLITS = {
    "keller": {"train": tuple(range(1, 17)), "val": tuple(range(17, 21)),
               "test": tuple(range(21, 26))},
    "official": {"train": (11, 12, 13, 14, 15, 16, 17, 18),
                 "val": (19, 20, 21, 23, 24, 25, 1, 4),
                 "test": (22, 2, 3, 5, 6, 7, 8, 9, 10)},
}
_LINE_RE = re.compile(r"^(?P<fname>\S+)\s+frames\s+(?P<ranges>[\d,\s\-]+)$")


def person_id(video):
    return int(video.split("_")[0][6:8])


def action_of(video):
    return video.split("_")[1]


def scenario_of(video):
    return video.split("_")[2]


class KTHStore:
    """All KTH frames and annotations, loaded once and shared by the splits."""

    def __init__(self, root, fg_thr=0.08, min_area=8):
        import h5py
        root = Path(root)
        h5 = root / "kth_frames.h5"
        txt = root / "00sequences.txt"
        if not h5.exists() or not txt.exists():
            raise FileNotFoundError(
                f"KTH not found under {root}: need kth_frames.h5 and 00sequences.txt "
                f"(python -m kth.prepare_kth).")
        with h5py.File(h5, "r") as f:
            self.frames = {k: f[k][:] for k in f.keys()}
        self.height, self.width = next(iter(self.frames.values())).shape[1:]
        self.annotations = self._parse(txt)
        self.fg_thr, self.min_area = fg_thr, min_area
        self.centroids = {k: self._track(v) for k, v in self.frames.items()}

    def _parse(self, txt):
        out = []
        for line in open(txt):
            m = _LINE_RE.match(line.strip())
            if not m:
                continue
            video = m.group("fname")
            if video not in self.frames:
                continue
            n = len(self.frames[video])
            for rng in m.group("ranges").split(","):
                rng = rng.strip()
                if not rng:
                    continue
                s, e = (map(int, rng.split("-")) if "-" in rng else (int(rng), int(rng)))
                out.append((video, s, min(e, n - 1)))
        return out

    def _track(self, frames):
        x = frames.astype(np.float32) / 255.0
        fg = np.abs(x - np.median(x, axis=0)) > self.fg_thr
        area = fg.sum(axis=(1, 2)).astype(np.float32)
        yy, xx = np.mgrid[0:x.shape[1], 0:x.shape[2]]
        cx = (fg * xx).sum(axis=(1, 2)) / np.maximum(area, 1)
        cy = (fg * yy).sum(axis=(1, 2)) / np.maximum(area, 1)
        c = np.stack([cx, cy], axis=-1).astype(np.float32)
        c[area < self.min_area] = np.nan
        return c

    def clips(self, split, scheme, min_frames):
        persons = set(SPLITS[scheme][split])
        return [(v, s, e) for v, s, e in self.annotations
                if e - s + 1 >= min_frames and person_id(v) in persons]


class KTHClips(Dataset):
    """One KTH split under one camera law."""

    def __init__(self, store, split="train", scheme="keller", camera=None, seq_len=16, step=2,
                 train=False, seed=0, eval_windows=1, resample_camera=False, camera_mix=1.0):
        self.store = store
        self.split, self.scheme = split, scheme
        self.seq_len, self.step = seq_len, step
        self.train = train
        self.camera = camera or CameraMotion("none", seq_len=seq_len)
        if self.camera.seq_len != seq_len:
            raise ValueError("camera.seq_len must equal seq_len")
        self.resample_camera = bool(resample_camera and train)
        self.clip_list = store.clips(split, scheme, seq_len * step)
        self.labels = [KTH_ACTIONS.index(action_of(v)) for v, _, _ in self.clip_list]
        n_clips = len(self.clip_list)
        self.windows = 1 if train else int(eval_windows)
        n_items = n_clips * self.windows
        split_salt = {"train": 1, "val": 2, "test": 3}[split]

        self.starts = None
        if not train:
            rng = np.random.RandomState(seed * 1009 + split_salt)
            self.starts = np.array([rng.randint(s, e - seq_len * step + 2)
                                    for v, s, e in self.clip_list
                                    for _ in range(self.windows)], dtype=np.int64)

        if self.camera.uses_keller_draws:
            per_clip = keller_constant_velocities(n_clips, self.camera.v_range)
            traj = np.repeat(per_clip[:, None, :], seq_len, axis=1)
        else:
            rng = np.random.RandomState(seed * 7919 + split_salt * 101 + 17)
            traj = np.stack([self.camera.draw(rng) for _ in range(n_clips)]) if n_clips else \
                np.zeros((0, seq_len, 2), dtype=np.int64)
        self.camera_mix = float(camera_mix)
        self.moving = np.ones(n_clips, dtype=bool)
        if self.camera_mix < 1.0:
            mix_rng = np.random.RandomState(seed * 7919 + split_salt * 101 + 29)
            self.moving = mix_rng.rand(n_clips) < self.camera_mix
            traj = traj * self.moving[:, None, None]
        self.trajectories = np.repeat(traj, self.windows, axis=0) if self.windows > 1 else traj
        assert len(self.trajectories) == n_items

    def __len__(self):
        return len(self.clip_list) * self.windows

    def clip_index(self, i):
        return i // self.windows

    def video(self, i):
        return self.clip_list[self.clip_index(i)][0]

    def __getitem__(self, i):
        video, s, e = self.clip_list[self.clip_index(i)]
        T, step = self.seq_len, self.step
        if self.train:
            start = np.random.randint(s, e - T * step + 2)
        else:
            start = int(self.starts[i])
        idx = start + step * np.arange(T)
        frames = self.store.frames[video][idx].astype(np.float32) / 255.0

        if self.resample_camera:
            v = (self.camera.draw(np.random) if np.random.rand() < self.camera_mix
                 else np.zeros((T, 2), dtype=np.int64))
        else:
            v = self.trajectories[i]
        d = displacements(v)
        clip = np.stack([np.roll(frames[t], (int(d[t, 1]), int(d[t, 0])), axis=(0, 1))
                         for t in range(T)])

        c = self.store.centroids[video][idx]
        person = np.full((T, 2), np.nan, dtype=np.float32)
        person[:-1] = c[1:] - c[:-1] + v[:-1]
        motion = v.astype(np.float32).copy()

        if self.train and np.random.rand() < 0.5:
            clip = clip[:, :, ::-1]
            motion[:, 0] *= -1
            person[:, 0] *= -1

        return (torch.from_numpy(np.ascontiguousarray(clip))[:, None],
                self.labels[self.clip_index(i)],
                torch.from_numpy(motion)[:, None, :],
                torch.from_numpy(person))
