import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import math
import wandb

def log_sequence_predictions(
        input_seq, target_seq, output_seq,
        split_name,
        num_samples=2,
        frames_per_row=10,
        upsample_scale=4,
        dpi=160
    ):
    batch_size = input_seq.size(0)
    num_samples = min(num_samples, batch_size)
    indices = np.random.choice(batch_size, num_samples, replace=False)

    input_len  = input_seq.size(1)
    target_len = target_seq.size(1)
    total_len  = input_len + target_len

    ncols = min(frames_per_row, total_len)
    nrows = math.ceil(total_len / ncols)

    fig_w = (ncols * upsample_scale) * 0.25
    fig_h = (3 * nrows * upsample_scale) * 0.25

    fig, axes = plt.subplots(
        3, num_samples,
        figsize=(fig_w * num_samples, fig_h),
        dpi=dpi,
        squeeze=False
    )

    for i, idx in enumerate(indices):
        s_in   = input_seq[idx].cpu()
        s_tgt  = target_seq[idx].cpu()
        s_pred = output_seq[idx].cpu()

        full_gt   = torch.cat([s_in,  s_tgt],  dim=0)
        full_pred = torch.cat([s_in,  s_pred], dim=0)
        full_diff = torch.cat([torch.zeros_like(s_in),
                               torch.abs(s_pred - s_tgt)], dim=0)

        for row, tensor, title in zip(
                range(3),
                (full_gt, full_pred, full_diff),
                ("Ground Truth", "Prediction", "Difference |Δ|")):
            grid = make_grid(
                tensor, nrow=ncols, normalize=True, padding=1
            )

            grid = F.interpolate(
                grid.unsqueeze(0),
                scale_factor=upsample_scale,
                mode='nearest'
            ).squeeze(0)

            axes[row, i].imshow(grid.permute(1, 2, 0).numpy(),
                                interpolation='nearest')
            axes[row, i].set_title(f"Sample {i+1} – {title}",
                                   fontsize=10)
            axes[row, i].axis('off')

    plt.tight_layout()
    wandb.log({f"{split_name}_sequences": wandb.Image(fig)})
    plt.close(fig)


def log_sequence_predictions_new(
    input_seq, target_seq, output_seq,
    split_name,
    num_samples: int = 4,
    vmax_diff: float = 1.0,
    subsample_t: int = 1,
    device: torch.device | None = None,
):
    T = target_seq.shape[1]
    T = T // subsample_t
    num_samples = min(num_samples, target_seq.shape[0])

    for idx in range(num_samples):
        gt_seq   = target_seq[idx].detach().cpu().squeeze()
        pred_seq = output_seq[idx].detach().cpu().squeeze()
        diff_seq = pred_seq - gt_seq

        fig_height = 3
        fig_width  = max(6, T)
        fig, axes  = plt.subplots(
            3, T,
            figsize=(fig_width, fig_height),
            gridspec_kw={"wspace": 0.005, "hspace": 0.03},
        )

        if T == 1:
            axes = axes.reshape(3, 1)

        for t in range(T):
            axes[0, t].imshow(gt_seq[t*subsample_t], cmap="gray", vmin=0, vmax=1)
            axes[1, t].imshow(pred_seq[t*subsample_t], cmap="gray", vmin=0, vmax=1)
            axes[2, t].imshow(
                diff_seq[t*subsample_t],
                cmap="bwr",
                vmin=-vmax_diff,
                vmax=vmax_diff,
            )

            for r in range(3):
                axes[r, t].axis("off")

        axes[0, 0].set_ylabel("GT",    rotation=0, labelpad=20, fontsize=10)
        axes[1, 0].set_ylabel("Pred",  rotation=0, labelpad=15, fontsize=10)
        axes[2, 0].set_ylabel("Error", rotation=0, labelpad=18, fontsize=10)

        fig.suptitle(f"{split_name} sample {idx}", fontsize=12)

        wandb.log({f"{split_name}sequence_{idx}": wandb.Image(fig)})
        plt.close(fig)


def log_state_evolution(
    h_states,
    gt_frames=None,
    split_name="train",
    num_samples=3,
    subsample_t=1,
    input_frames=None,
    motion=None,
    v_list=None,
):
    B, T, K, H, W = h_states.shape
    num_samples = min(num_samples, B)
    indices = np.random.choice(B, num_samples, replace=False)

    T_shown = max(1, T // subsample_t)
    has_gt_frames = gt_frames is not None
    select_slots = motion is not None and v_list is not None
    v_index = {tuple(v): k for k, v in enumerate(v_list)} if select_slots else None

    for idx in indices:
        h_sample = h_states[idx].detach().cpu()
        gt_sample = gt_frames[idx].detach().cpu() if has_gt_frames else None

        if select_slots:
            window = motion[idx, :input_frames].reshape(-1, 2).tolist()
            observed = {tuple(int(x) for x in v) for v in window}
            slot_idx = sorted(v_index[v] for v in observed if v in v_index)
            if not slot_idx:
                slot_idx = list(range(K))
            h_sample = h_sample[:, slot_idx]
            slot_labels = [f"v={v_list[k]}" for k in slot_idx]
        else:
            slot_labels = [f"h slot{k}" for k in range(K)]

        rows_k = h_sample.shape[1]
        rows = rows_k + int(has_gt_frames)
        gt_row = rows_k

        fig, axes = plt.subplots(
            rows, T_shown,
            figsize=(max(6, T_shown * 1.3), max(2, rows * 1.3) + 0.6),
            gridspec_kw={"wspace": 0.05, "hspace": 0.35},
            squeeze=False,
        )

        for tt in range(T_shown):
            t = tt * subsample_t
            is_decoder = input_frames is not None and t >= input_frames
            title_color = "crimson" if is_decoder else "black"

            for k in range(rows_k):
                v = h_sample[t, k]
                vmax = v.abs().max().clamp(min=1e-8).item()
                axes[k, tt].imshow(v, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                axes[k, tt].axis("off")

            if has_gt_frames:
                axes[gt_row, tt].imshow(
                    gt_sample[t].mean(dim=0), cmap="gray", vmin=0, vmax=1
                )
                axes[gt_row, tt].axis("off")

            axes[0, tt].set_title(f"t={t}", fontsize=8, color=title_color)

        for k in range(rows_k):
            axes[k, 0].text(-0.4, 0.5, slot_labels[k], rotation=90,
                             va="center", ha="center", fontsize=8,
                             transform=axes[k, 0].transAxes)
        if has_gt_frames:
            axes[gt_row, 0].text(-0.4, 0.5, "frame\n(GT)", rotation=90,
                                  va="center", ha="center", fontsize=7,
                                  transform=axes[gt_row, 0].transAxes)

        title = f"{split_name} h evolution — sample {idx}"
        if input_frames is not None:
            title += " (red titles = decoder)"
        fig.suptitle(title, fontsize=10)

        wandb.log({f"{split_name}_states_sample{idx}": wandb.Image(fig)})
        plt.close(fig)


def log_velocity_report(summary, split_name="train", epoch=None):
    if summary is None:
        return

    table = wandb.Table(columns=["t", "acc_pct", "acc_pct_seq", "binding_loss",
                                 "mean_l2", "mean_l2_seq", "correct", "total"])
    for t in range(summary["T"]):
        table.add_data(
            t + 1,
            summary["per_t_acc_stepwise"][t],
            summary["per_t_acc"][t],
            summary["per_t_binding_loss"][t],
            summary["per_t_l2_stepwise"][t],
            summary["per_t_l2"][t],
            summary["per_t_correct_stepwise"][t],
            summary["per_t_total"][t],
        )

    log_dict = {
        f"{split_name}_vel_table"       : table,
        f"{split_name}_vel_overall_acc" : summary["overall_acc_stepwise"],
        f"{split_name}_vel_overall_l2"  : summary["overall_l2_stepwise"],
        f"{split_name}_vel_encoder_acc" : summary["encoder_acc_stepwise"],
        f"{split_name}_vel_decoder_acc" : summary["decoder_acc_stepwise"],
        f"{split_name}_vel_last_step_acc": summary["last_step_acc_stepwise"],
        f"{split_name}_vel_last_step_l2" : summary["last_step_l2_stepwise"],
        f"{split_name}_vel_bootstrap_acc": summary["per_t_acc_stepwise"][0],
        f"{split_name}_vel_overall_acc_seq" : summary["overall_acc"],
        f"{split_name}_vel_encoder_acc_seq" : summary["encoder_acc"],
        f"{split_name}_vel_decoder_acc_seq" : summary["decoder_acc"],
    }
    if epoch is not None:
        log_dict["epoch"] = epoch

    wandb.log(log_dict)

def log_motion_classification_states(
    h_states,
    frames,
    mask_track=None,
    velocities=None,
    v_list=None,
    gt_motion=None,
    split_name="val",
    epoch=None,
    step=None,
    num_samples=2,
    subsample_t=1,
    max_slots=6,
    save_dir=None,
    dpi=160,
    show_shape_readout=False,
    mask_color="#E69F00",
    mask_lw=1.2,
    mask_halo=1.6,
    mask_halo_color="black",
):
    payload = {}
    B, T, V, H, W = h_states.shape
    n = min(num_samples, B)
    steps = list(range(0, T, max(1, subsample_t)))
    if steps[-1] != T - 1:
        steps.append(T - 1)

    has_lattice = v_list is not None and len(v_list) > 1

    if save_dir is not None:
        from pathlib import Path as _Path
        save_dir = _Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        h = h_states[i].detach().cpu()
        fr = frames[i].detach().cpu()
        mk = mask_track[i].detach().cpu() if mask_track is not None else None

        v_fig = v_bg = None
        if gt_motion is not None:
            g = gt_motion[i].detach().cpu()
            v_fig = tuple(int(x) for x in g[-2, 0])
            v_bg = tuple(int(x) for x in g[-2, -1])

        missing = []
        fig_row = None
        if has_lattice:
            index = {tuple(v): k for k, v in enumerate(v_list)}
            chosen = []
            for vel in (v_fig, v_bg):
                if vel in index and index[vel] not in chosen:
                    if vel == v_fig:
                        fig_row = len(chosen)
                    chosen.append(index[vel])
                elif vel is not None:
                    missing.append(str(vel))
            for k in range(len(v_list)):
                if len(chosen) >= max_slots:
                    break
                if k not in chosen:
                    chosen.append(k)
        elif velocities is not None:
            vl = velocities[i, -1].detach().cpu().round().long()
            chosen = list(range(min(V, max_slots)))
            for k in chosen:
                if tuple(int(x) for x in vl[k]) == v_fig:
                    fig_row = k
                    break
        else:
            chosen, fig_row = [0], 0
        labels = [f"$s_{{{k}}}$" for k in range(len(chosen))]

        def local_var(x, k=3):
            x = x[None, None]
            pad = k // 2
            mu = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(x, (pad,) * 4, mode="circular"), k, stride=1)
            mu2 = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(x * x, (pad,) * 4, mode="circular"), k, stride=1)
            return (mu2 - mu * mu).clamp(min=0)[0, 0]

        add_readout = show_shape_readout and fig_row is not None

        rows = len(chosen) + int(add_readout) + 1
        readout_row = len(chosen)
        frame_row = readout_row + int(add_readout)

        fig_w = max(7, len(steps) * 0.92)
        fig, axes = plt.subplots(
            rows, len(steps),
            figsize=(fig_w, rows * 1.02 + 0.85),
            gridspec_kw={"wspace": 0.035, "hspace": 0.16},
            squeeze=False,
        )

        all_labels = [ln for lab in labels for ln in lab.split("\n")]
        all_labels += ["input", "+ GT mask"]
        longest = max(len(x.replace("$", "")) for x in all_labels)
        left_margin = min(0.30, (longest * 0.078 + 0.30) / fig_w)

        for col, t in enumerate(steps):
            for r, k in enumerate(chosen):
                m = h[t, k]
                lim = m.abs().max().clamp(min=1e-8).item()
                axes[r, col].imshow(m, cmap="coolwarm", vmin=-lim, vmax=lim,
                                    interpolation="nearest")
            if add_readout:
                axes[readout_row, col].imshow(local_var(h[t, chosen[fig_row]]),
                                              cmap="magma", interpolation="nearest")
            ax = axes[frame_row, col]
            ax.imshow(fr[t].mean(0), cmap="gray", interpolation="nearest")
            if mk is not None:
                m2 = mk[t].amax(0).numpy()
                pad = 3
                mp = np.pad(m2, pad, mode="wrap")
                Hm, Wm = m2.shape
                cs = ax.contour(mp, levels=[0.5], colors=[mask_color],
                                linewidths=mask_lw, antialiased=True,
                                origin="upper",
                                extent=(-pad - 0.5, Wm + pad - 0.5,
                                        Hm + pad - 0.5, -pad - 0.5))
                if mask_halo:
                    fx = [patheffects.withStroke(linewidth=mask_lw + mask_halo,
                                                 foreground=mask_halo_color,
                                                 alpha=0.9)]
                    for coll in (cs.collections if hasattr(cs, "collections") else [cs]):
                        coll.set_path_effects(fx)
                ax.set_xlim(-0.5, Wm - 0.5)
                ax.set_ylim(Hm - 0.5, -0.5)
            axes[0, col].set_title(f"$t$ = {t}", fontsize=11, pad=5)
            for r in range(rows):
                axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
                for sp in axes[r, col].spines.values():
                    sp.set_linewidth(0.4); sp.set_color("0.75")

        def ylabel(row, text):
            axes[row, 0].set_ylabel(text, fontsize=11, rotation=0, ha="right",
                                    va="center", labelpad=8)

        for r, lab in enumerate(labels):
            ylabel(r, lab)
        if add_readout:
            ylabel(readout_row, f"local var\n$s_{{{fig_row}}}$")
        ylabel(frame_row, "input\n+ GT mask" if mk is not None else "input")

        what = "hidden state over time" if not has_lattice and velocities is None \
            else "hidden state per velocity copy"
        title = f"{split_name} — {what}, sample {i}"
        if epoch is not None:
            title += f"   (epoch {epoch})"
        if missing:
            title += f"\nvelocities off this model's lattice: {', '.join(missing)}"
        fig.suptitle(title, fontsize=12, y=0.995)
        fig.subplots_adjust(top=0.90 - 0.02 * (title.count(chr(10))),
                            bottom=0.015, left=left_margin, right=0.995)

        if save_dir is not None:
            stem = f"{split_name}_states_sample{i}"
            fig.savefig(save_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
            fig.savefig(save_dir / f"{stem}.pdf", bbox_inches="tight")

        payload[f"{split_name}_states_sample{i}"] = wandb.Image(fig)
        plt.close(fig)

    if payload:
        wandb.log(payload, step=step) if step is not None else wandb.log(payload)
