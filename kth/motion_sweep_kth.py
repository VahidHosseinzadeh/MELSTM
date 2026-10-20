#!/usr/bin/env python
"""Score trained KTH checkpoints with the camera moving under the Moving MNIST motion laws."""
import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kth.camera_motion import CameraMotion
from kth.kth_dataset import KTHClips, KTHStore
from kth.sweep_kth import load_run
from kth.train_kth import class_report, run_epoch
from motion_difficulty_sweep import (NAMED_REGIMES, RATE_TOL, REGIME_CHECKS,
                                     SPEED_TOL)
from motion_factors_sweep import (D_GRID, S_GRID, TARGET_RATE, TRAIN_N,
                                  TRAIN_S, compensated_p)

SPEED_GRID = (1, 2, 3, 4, 5)

_REGIME_KEYS = {"motion_mode": "mode", "transition_mode": "transition",
                "min_segment": "min_segment", "max_segment": "max_segment",
                "smooth_probability": "smooth_probability", "p_change": "p_change"}


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run names (models/<run>_best.pth)")
    p.add_argument("--save_dir", default="./experiments_kth")
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "data" / "kth"))
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--stats_draws", type=int, default=20000,
                   help="trajectories per motion law for its realised statistics")
    p.add_argument("--max_clips", type=int, default=None,
                   help="score only the first N test clips (quick checks)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="./experiments_kth/motion_sweep")
    return p.parse_args(argv)


def train_law(cfg):
    if cfg.get("camera", "none") != "shake" or cfg.get("camera_mix", 1.0) < 1.0:
        raise SystemExit(f"{cfg.get('run_name')}: the 'train' cell expects a run trained on "
                         f"the shake alone, not camera={cfg.get('camera')!r} "
                         f"(mix {cfg.get('camera_mix', 1.0)})")
    return dict(mode="shake", shake_amp=tuple(cfg["shake_amp"]),
                shake_period=tuple(cfg["shake_period"]), shake_axes=cfg["shake_axes"],
                shake_vmax=cfg["shake_vmax"])


def cells(cfg):
    mm = dict(v_range=TRAIN_N, neighbor_kernel="symmetric", include_zero=False)
    out = [("none", "reference", 0.0, dict(mode="none")),
           ("train", "reference", float("nan"), train_law(cfg))]
    for d in D_GRID:
        out.append((f"A:d={d:.2f}", "rate", float(d),
                    dict(mm, mode="stochastic", transition="smooth", p_change=d ** 2.170,
                         smooth_probability=TRAIN_S)))
    for s in S_GRID:
        out.append((f"B:s={s:.2f}", "jump", float(s),
                    dict(mm, mode="stochastic", transition="smooth",
                         p_change=compensated_p(TARGET_RATE, s, TRAIN_N),
                         smooth_probability=s)))
    for name, kw in NAMED_REGIMES.items():
        unknown = set(kw) - set(_REGIME_KEYS)
        if unknown:
            raise SystemExit(f"regime {name}: no camera equivalent for {sorted(unknown)}")
        out.append((f"C:{name}", "named", float("nan"),
                    dict(mm, **{_REGIME_KEYS[k]: v for k, v in kw.items()})))
    for R in SPEED_GRID:
        out.append((f"E:R={R}", "speed", float(R), dict(mode="constant", v_range=R)))
    return out


def law_stats(v):
    u = np.asarray(v, np.int64)[:, :-1]
    dv = np.abs(u[:, 1:] - u[:, :-1]).max(-1)
    changed = dv > 0
    R = int(np.abs(u).max())
    K = (2 * R + 1) ** 2
    code = (u[..., 0] + R) * (2 * R + 1) + (u[..., 1] + R)
    joint = np.bincount((code[:, :-1] * K + code[:, 1:]).ravel(), minlength=K * K)
    P = joint / joint.sum()
    prev = P.reshape(K, K).sum(1)
    h = lambda q: float(-(q[q > 0] * np.log2(q[q > 0])).sum())
    d = np.cumsum(u, axis=1)
    return dict(rate=float(changed.mean()),
                jump=float(dv[changed].mean()) if changed.any() else 0.0,
                entropy=h(P) - h(prev),
                speed=float(np.abs(u).max(-1).mean()),
                mean_speed=float(np.sqrt((u ** 2).sum(-1)).mean()),
                net_disp=float(np.sqrt((d[:, -1] ** 2).sum(-1)).mean()),
                travel=float(np.abs(d[:, -1]).max(-1).mean()),
                excursion=float(np.abs(d).max(-1).max(-1).mean()))


def check_regimes(stats):
    ok = True
    for name, want in REGIME_CHECKS.items():
        got = stats[f"C:{name}"]
        bad = (abs(got["rate"] - want["rate"]) > RATE_TOL
               or abs(got["mean_speed"] - want["speed"]) > SPEED_TOL)
        ok &= not bad
        print(f"  check {name:10s} rate {got['rate']:.3f} (Moving MNIST {want['rate']:.3f})  "
              f"|v|_2 {got['mean_speed']:.3f} ({want['speed']:.3f})  {'FAIL' if bad else 'ok'}")
    return ok


@torch.no_grad()
def score_run(run, sets, a, device):
    model, cfg, epoch = load_run(run, a.save_dir, device)
    criterion = nn.CrossEntropyLoss()
    out, labels = {}, None
    for name, ds in sets.items():
        loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
        r = run_epoch(model, loader, device, criterion, collect=True)
        y_true, y_pred = r.pop("_y_true"), r.pop("_y_pred")
        r.pop("_global_step", None)
        rep = class_report(y_true, y_pred)
        out[name] = {**{k: v for k, v in r.items() if isinstance(v, (int, float))},
                     "acc_locomotion": rep["acc_locomotion"], "acc_inplace": rep["acc_inplace"],
                     "pred": [int(q) for q in y_pred]}
        labels = labels or [int(t) for t in y_true]
    return cfg, epoch, out, labels


def main(argv=None):
    a = get_args(argv)
    device = torch.device(a.device) if a.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = KTHStore(a.root)

    cfgs = {run: torch.load(Path(a.save_dir) / "models" / f"{run}_best.pth", map_location="cpu",
                            weights_only=False)["config"] for run in a.runs}
    laws = {json.dumps(train_law(c), sort_keys=True) for c in cfgs.values()}
    if len(laws) > 1:
        raise SystemExit(f"the runs were trained on different shakes: {sorted(laws)}")
    first = next(iter(cfgs.values()))
    T = int(first["seq_len"])

    todo = cells(first)
    sets, meta, trajectories = {}, {}, {}
    rng_stats = np.random.RandomState(12345)
    for name, axis, level, kw in todo:
        cam = CameraMotion(seq_len=T, **kw)
        ds = KTHClips(store, "test", scheme=first["split_scheme"], camera=cam, seq_len=T,
                      step=int(first["step"]), seed=a.data_seed)
        trajectories[name] = ds.trajectories.astype(np.int8)
        if a.max_clips:
            ds = torch.utils.data.Subset(ds, list(range(min(a.max_clips, len(ds)))))
        sets[name] = ds
        stats = law_stats(np.stack([cam.draw(rng_stats) for _ in range(a.stats_draws)]))
        meta[name] = {"axis": axis, "level": level, "law": cam.describe(),
                      "kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in kw.items()},
                      "stats": stats, "test_stats": law_stats(trajectories[name])}
    n_clips = len(next(iter(sets.values())))
    print(f"{len(sets)} cells x {n_clips} clips, device {device}; motion laws:")
    for name, m in meta.items():
        s = m["stats"]
        print(f"  {name:20s} rate {s['rate']:.3f}  jump {s['jump']:.2f}  H {s['entropy']:.2f} "
              f"bits  speed {s['speed']:.2f}  travel {s['travel']:5.1f} px   {m['law']}")
    if not check_regimes({k: m["stats"] for k, m in meta.items()}):
        raise SystemExit("the named regimes do not reproduce Moving MNIST's statistics")

    np.savez_compressed(out / "motion_cells.npz", **{k.replace(":", "_").replace("=", "_"): v
                                                     for k, v in trajectories.items()})
    labels_all = None
    for run in a.runs:
        cfg, epoch, res, labels = score_run(run, sets, a, device)
        labels_all = labels_all or labels
        line = "  ".join(f"{k} {res[k]['acc']:.3f}" for k in ("none", "train", "E:R=1", "E:R=2"))
        print(f"{run:28s} best epoch {epoch:3d}  {line}")
        with open(out / f"motion_sweep_{run}.json", "w") as f:
            json.dump({"run": run, "config": cfg, "epoch": epoch, "cells": res}, f)
    with open(out / "motion_cells.json", "w") as f:
        json.dump({"cells": meta, "labels": labels_all, "data_seed": a.data_seed,
                   "n_clips": n_clips, "seq_len": T,
                   "npz_keys": {k: k.replace(":", "_").replace("=", "_") for k in meta}},
                  f, indent=1)
    print(f"wrote {out}/motion_sweep_*.json, motion_cells.json, motion_cells.npz")


if __name__ == "__main__":
    main()
