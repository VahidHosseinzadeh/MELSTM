"""Re-evaluate best checkpoints on the fixed Moving MNIST test set (MSE and L1 separately)."""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Subset

from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset
from train_eval_utils import build_model, eval_epoch


class SplitLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = []

    def forward(self, out, tgt):
        mse, l1 = F.mse_loss(out, tgt), F.l1_loss(out, tgt)
        self.rows.append((mse.item(), l1.item(), out.size(0)))
        return mse + l1


def test_dataset(cfg, root):
    return TDMovingMNISTDataset(
        root=root, train=False, seq_len=cfg["seq_len"], num_digits=2,
        image_size=cfg["image_size"], max_speed=cfg["data_v_range"],
        motion_mode=cfg["motion_mode"], transition_mode=cfg["transition_mode"],
        min_segment=cfg["min_segment"], max_segment=cfg["max_segment"],
        p_change=cfg["p_change"], smooth_probability=cfg["smooth_probability"],
        motion_difficulty=None, freeze_after=cfg["input_frames"],
        min_center_distance=20, reject_overlap=True, require_distinct_velocities=True,
        return_motion=bool(cfg.get("check_velocity_predictor") or cfg.get("show_h_state")),
        return_positions=False, transform=None, download=True,
        random=False, seed=123, max_tries=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--root", default="./data")
    p.add_argument("--limit", type=int, default=0, help="smoke test: only the first N sequences")
    args = p.parse_args()

    wandb.init(mode="disabled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = json.load(open(args.manifest))
    done = {}
    if os.path.exists(args.out):
        done = {r["name"]: r for r in json.load(open(args.out))}
    results = list(done.values())
    print(f"device {device}; {len(runs)} runs ({len(done)} already in {args.out})", flush=True)
    for e in runs:
        if e["name"] in done:
            continue
        t0 = time.time()
        hist = json.load(open(e["history"]))
        cfg = hist["config"]
        ds = test_dataset(cfg, args.root)
        if args.limit:
            ds = Subset(ds, list(range(args.limit)))
        base = ds.dataset if isinstance(ds, Subset) else ds
        base.reset_rng()
        loader = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                            pin_memory=(device.type == "cuda"), persistent_workers=False)
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(e["ckpt"], map_location=device))
        model.eval()
        crit = SplitLoss()
        total = eval_epoch(model, loader, crit, device, cfg["input_frames"], 0, split_name="test")
        n = sum(r[2] for r in crit.rows)
        mse = sum(r[0] * r[2] for r in crit.rows) / n
        l1 = sum(r[1] * r[2] for r in crit.rows) / n
        logged = hist["history"]["test_loss"][-1]
        rec = dict(e, n=n, test_mse=mse, test_l1=l1, test_loss_recomputed=total,
                   test_loss_logged=logged, seconds=time.time() - t0)
        results.append(rec)
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"{e['name']:28s} n={n:5d}  MSE {mse:.4e}  L1 {l1:.4e}  MSE+L1 {total:.4e} "
              f"(logged {logged:.4e}, diff {total - logged:+.1e})  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
