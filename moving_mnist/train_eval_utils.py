import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from moving_mnist_dataset import FixedVelocityMovingMNIST
from visualization import (
    log_sequence_predictions,
    log_sequence_predictions_new,
    log_state_evolution,
    log_velocity_report,
)
from velocity_metrics import VelocityMetrics
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM
from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM


def build_model(cfg):
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))

    name = get("model")
    hidden = get("hidden_size")
    kernel = get("kernel_size", 3)
    dec_layers = get("decoder_conv_layers", 1)
    dec_channels = get("decoder_hidden_size") or hidden
    detach_feedback = not get("no_detach")
    decoder_act = get("decoder_act") or "relu"

    if name in ("felstm", "lstm"):
        v_range = get("v_range", 0) if name == "felstm" else 0
        if name == "lstm":
            assert get("v_range", 0) == 0, "v_range must be 0 for lstm"
        if get("decoder_skip"):
            raise ValueError("--decoder_skip exists for melstm only")
        return Seq2SeqFEConvLSTM(
            input_channels=1, hidden_channels=hidden, kernel_size=kernel,
            v_range=v_range, pool_type="max",
            decoder_conv_layers=dec_layers, decoder_channels=dec_channels,
            detach_feedback=detach_feedback, decoder_act=decoder_act)

    if name == "melstm":
        prior_w = get("track_prior_weight")
        return Seq2SeqMEConvLSTM(
            input_channels=1, hidden_channels=hidden, kernel_size=kernel,
            n_slots=get("num_vel_modes", 2), slot_reduce="max",
            decoder_layers=dec_layers, decoder_channels=dec_channels,
            track_mode=get("track_mode") or "h",
            track_prior_weight=0.5 if prior_w is None else prior_w,
            n_proposals=get("n_proposals"),
            track_window=get("track_window") or 2,
            detach_feedback=detach_feedback, decoder_act=decoder_act,
            decoder_skip=bool(get("decoder_skip")))

    raise ValueError(f"unknown model {name!r}")


def _unpack_batch(batch, device):
    seq = batch[0].to(device)
    gt_motion = batch[2].to(device) if len(batch) > 2 else None
    return seq, gt_motion


def _run_model(model, input_seq, pred_len, target_seq,
               want_velocity, want_states, track_decoder_velocity=None):
    if isinstance(model, Seq2SeqMEConvLSTM):
        if track_decoder_velocity is None:
            track_decoder_velocity = model.training
        result = model(
            input_seq,
            pred_len=pred_len,
            target_seq=target_seq,
            track_decoder_velocity=track_decoder_velocity,
            return_velocity=want_velocity,
            return_states=want_states,
        )
        if not (want_velocity or want_states):
            return result, None, None
        result = list(result if isinstance(result, tuple) else [result])
        output_seq = result.pop(0)
        pred_motion = result.pop(0) if want_velocity else None
        states = result.pop(0) if want_states else None
        return output_seq, pred_motion, states

    if isinstance(model, Seq2SeqFEConvLSTM):
        if not want_states:
            return model(input_seq, pred_len=pred_len), None, None
        output_seq, states = model(input_seq, pred_len=pred_len, return_states=True)
        return output_seq, None, states

    return model(input_seq, pred_len=pred_len), None, None


def _velocity_flags(model, gt_motion, show_h_state=False):
    is_melstm = isinstance(model, Seq2SeqMEConvLSTM)
    is_felstm = isinstance(model, Seq2SeqFEConvLSTM)
    want_velocity = is_melstm and gt_motion is not None
    want_states = want_velocity or (is_felstm and show_h_state and gt_motion is not None)
    return want_velocity, want_states


class ValCurveRecorder:
    def __init__(self, val_dataset, n_sequences, interval, input_frames,
                 device, batch_size=64):
        n = min(n_sequences, len(val_dataset))
        self.data = torch.stack([val_dataset[i][0] for i in range(n)])
        self.interval = interval
        self.input_frames = input_frames
        self.device = device
        self.batch_size = batch_size
        self.step = 0
        self.train_steps, self.train_losses = [], []
        self.val_steps, self.val_means, self.val_stds = [], [], []

    @torch.no_grad()
    def _val_loss(self, model):
        was_training = model.training
        model.eval()
        per_seq = []
        for s in range(0, self.data.size(0), self.batch_size):
            seq = self.data[s:s + self.batch_size].to(self.device)
            inp = seq[:, :self.input_frames]
            tgt = seq[:, self.input_frames:]
            pred, _, _ = _run_model(model, inp, tgt.size(1), None, False, False)
            d = pred - tgt
            per_seq.append((d.pow(2).mean(dim=(1, 2, 3, 4))
                            + d.abs().mean(dim=(1, 2, 3, 4))).cpu())
        if was_training:
            model.train()
        per_seq = torch.cat(per_seq)
        return per_seq.mean().item(), per_seq.std().item()

    def on_batch(self, model, train_loss):
        self.step += 1
        self.train_steps.append(self.step)
        self.train_losses.append(float(train_loss))
        if self.interval > 0 and self.step % self.interval == 0:
            m, s = self._val_loss(model)
            self.val_steps.append(self.step)
            self.val_means.append(m)
            self.val_stds.append(s)
            wandb.log({"val_curve_loss": m, "val_curve_std": s,
                       "global_batch": self.step})

    def as_dict(self):
        return {
            "train_curve": {"step": self.train_steps,
                            "loss": self.train_losses},
            "val_curve": {"step": self.val_steps,
                          "loss_mean": self.val_means,
                          "loss_std": self.val_stds},
        }

    def state_dict(self):
        return {
            "data": self.data,
            "step": self.step,
            "train_steps": self.train_steps,
            "train_losses": self.train_losses,
            "val_steps": self.val_steps,
            "val_means": self.val_means,
            "val_stds": self.val_stds,
        }

    def load_state_dict(self, state):
        self.data = state["data"]
        self.step = state["step"]
        self.train_steps = list(state["train_steps"])
        self.train_losses = list(state["train_losses"])
        self.val_steps = list(state["val_steps"])
        self.val_means = list(state["val_means"])
        self.val_stds = list(state["val_stds"])


def train_epoch(model, dataloader, optimizer, criterion, device, input_frames, grad_clip=None,
                curve_recorder=None, show_h_state=False, lr_fn=None, step_offset=0, diag_every=0):
    model.train()
    running_loss = 0.0
    relus = ([mod for mod in model.modules() if isinstance(mod, (torch.nn.ReLU, torch.nn.LeakyReLU))]
             if diag_every else [])
    relu_frac = []
    def _relu_hook(mod, inp, out):
        if relu_frac is not None and diag_on[0]:
            relu_frac.append((out > 0).float().mean().item())
    diag_on = [False]
    hooks = [mod.register_forward_hook(_relu_hook) for mod in relus]
    gnorms = []
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False

    pbar = tqdm(dataloader, desc="Training", leave=False, disable=True)
    for i, batch in enumerate(pbar):
        seq, gt_motion = _unpack_batch(batch, device)
        input_seq = seq[:, :input_frames]
        target_seq = seq[:, input_frames:]
        pred_len = target_seq.size(1)

        want_velocity, want_states = _velocity_flags(model, gt_motion, show_h_state)
        want_states = want_states and i == 0

        step = step_offset + i
        if lr_fn is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_fn(step)
        diag_on[0] = bool(diag_every) and (step + 1) % diag_every == 0

        optimizer.zero_grad()
        output_seq, pred_motion, states = _run_model(
            model, input_seq, pred_len, target_seq, want_velocity, want_states
        )
        loss = criterion(output_seq, target_seq)
        loss.backward()

        if grad_clip is not None:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if diag_every:
                gnorms.append(float(gnorm))

        optimizer.step()

        if diag_on[0]:
            payload = {"lr_step": optimizer.param_groups[0]["lr"], "global_batch": step + 1}
            if gnorms:
                payload.update(grad_norm_mean=sum(gnorms) / len(gnorms), grad_norm_max=max(gnorms))
            if relu_frac:
                payload["relu_active_frac"] = min(relu_frac)
            wandb.log(payload)
            gnorms.clear(); relu_frac.clear()
            diag_on[0] = False

        if want_velocity:
            velocity_metrics.update(pred_motion, gt_motion)
            has_velocity_data = True

        batch_loss = loss.item()
        running_loss += batch_loss * seq.size(0)
        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        if curve_recorder is not None:
            curve_recorder.on_batch(model, batch_loss)

        if i == 0:
            log_sequence_predictions(input_seq, target_seq, output_seq, split_name="train")
            if states is not None:
                log_state_evolution(
                    states["h"],
                    gt_frames=torch.cat([input_seq, target_seq], dim=1),
                    split_name="train",
                    input_frames=input_frames,
                    motion=gt_motion,
                    v_list=model.cell.v_list if isinstance(model, Seq2SeqFEConvLSTM) else None,
                )

    for h in hooks:
        h.remove()

    if has_velocity_data:
        velocity_metrics.report("Training Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name="train")

    return running_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, criterion, device, input_frames, epoch, split_name,
               decoder_velocity_mode="frozen", show_h_state=False):
    model.eval()
    track = decoder_velocity_mode == "tracked"
    running_loss = 0.0
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False, disable=True)
        for i, batch in enumerate(pbar):
            seq, gt_motion = _unpack_batch(batch, device)
            input_seq = seq[:, :input_frames]
            target_seq = seq[:, input_frames:]
            pred_len = target_seq.size(1)

            want_velocity, want_states = _velocity_flags(model, gt_motion, show_h_state)
            want_states = want_states and i == 0

            output_seq, pred_motion, states = _run_model(
                model, input_seq, pred_len, target_seq, want_velocity, want_states,
                track_decoder_velocity=True if track else None,
            )
            loss = criterion(output_seq, target_seq)
            batch_loss = loss.item()
            running_loss += batch_loss * seq.size(0)
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            if want_velocity:
                velocity_metrics.update(pred_motion, gt_motion)
                has_velocity_data = True

            if i == 0:
                log_sequence_predictions(input_seq, target_seq, output_seq, split_name=split_name)
                if states is not None:
                    log_state_evolution(
                        states["h"],
                        gt_frames=torch.cat([input_seq, target_seq], dim=1),
                        split_name=split_name,
                        input_frames=input_frames,
                        motion=gt_motion,
                        v_list=model.cell.v_list if isinstance(model, Seq2SeqFEConvLSTM) else None,
                    )

    if has_velocity_data:
        velocity_metrics.report(f"{split_name} Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name=split_name, epoch=epoch)

    return running_loss / len(dataloader.dataset)


def eval_len_generalization(model, dataloader, device, input_frames, subsample_t=1,
                            n_strip_sequences=3, show_h_state=False):
    model.eval()
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False
    first_pass = True
    per_seq_chunks = []
    baseline_chunks = []
    age_chunks = []
    strip = {}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False, disable=True)
        for batch in pbar:
            seq, gt_motion = _unpack_batch(batch, device)
            inp, tgt = seq[:, :input_frames], seq[:, input_frames:]
            T = tgt.size(1)

            baseline_chunks.append(
                ((inp[:, -1:] - tgt) ** 2).mean(dim=(2, 3, 4)).detach().cpu()
            )

            if gt_motion is not None:
                f = input_frames
                m = gt_motion[:, :f - 1]
                changed = (m[:, 1:] != m[:, :-1]).any(dim=-1)
                idx = torch.arange(1, f - 1, device=changed.device).view(1, -1, 1)
                t_last = torch.where(changed, idx,
                                     torch.zeros_like(idx)).amax(dim=1)
                age_chunks.append(((f - 1) - t_last).detach().cpu())

            want_velocity, want_states = _velocity_flags(model, gt_motion, show_h_state)
            want_states = want_states and first_pass

            pred, pred_motion, states = _run_model(
                model, inp, T, None, want_velocity, want_states
            )

            if want_velocity:
                velocity_metrics.update(pred_motion, gt_motion)
                has_velocity_data = True

            per_ex_t = ((pred - tgt) ** 2).mean(dim=(2, 3, 4))
            per_seq_chunks.append(per_ex_t.detach().cpu())

            if first_pass:
                first_pass = False
                n = n_strip_sequences
                strip = {
                    "strip_input": inp[:n].detach().cpu().numpy(),
                    "strip_gt":    tgt[:n].detach().cpu().numpy(),
                    "strip_pred":  pred[:n].detach().cpu().numpy(),
                }

                log_sequence_predictions_new(
                    inp, tgt, pred, split_name="len_gen",
                    num_samples=10, device=device, subsample_t=subsample_t,
                )
                if states is not None:
                    log_state_evolution(
                        states["h"],
                        gt_frames=torch.cat([inp, tgt], dim=1),
                        split_name="len_gen",
                        subsample_t=subsample_t,
                        input_frames=input_frames,
                        motion=gt_motion,
                        v_list=model.cell.v_list if isinstance(model, Seq2SeqFEConvLSTM) else None,
                    )

            pbar.set_postfix({"loss": per_ex_t.mean().item()})

    if has_velocity_data:
        velocity_metrics.report("Length Generalization Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name="len_gen")

    per_seq_err = torch.cat(per_seq_chunks, dim=0).numpy()
    mean = per_seq_err.mean(axis=0)
    std = per_seq_err.std(axis=0)

    details = {
        "per_seq_err": per_seq_err,
        "per_seq_err_copy_last": torch.cat(baseline_chunks, dim=0).numpy(),
        **strip,
    }
    if age_chunks:
        details["boundary_age"] = torch.cat(age_chunks, dim=0).numpy()
    return mean, std, details


def eval_velocity_generalization(model, device, args):
    vx_vals = np.arange(args.gen_vel_min, args.gen_vel_max + 1, args.gen_vel_step)
    vy_vals = np.arange(args.gen_vel_min, args.gen_vel_max + 1, args.gen_vel_step)
    err_mat = np.zeros((len(vy_vals), len(vx_vals)))

    crit = torch.nn.MSELoss(reduction='none')
    model.eval()

    vel_pbar = tqdm(total=len(vy_vals)*len(vx_vals), desc="Evaluating Velocity Generalization", leave=False, disable=True)
    for iy, vy in enumerate(vy_vals):
        for ix, vx in enumerate(vx_vals):
            vel_pbar.set_postfix({"vx": vx, "vy": vy})
            dataset = FixedVelocityMovingMNIST(
                vx=vx, vy=vy,
                root=args.root,
                train=False,
                seq_len=args.seq_len,
                image_size=args.image_size,
                num_digits=2,
                random=False)

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            mse_sum, n_seen = 0.0, 0
            with torch.no_grad():
                batch_pbar = tqdm(loader, desc=f"vx={vx}, vy={vy}", leave=False, disable=True)
                for seq, _ in batch_pbar:
                    seq = seq.to(device)
                    inp, tgt = seq[:, :args.input_frames], seq[:, args.input_frames:]
                    pred, _, _ = _run_model(model, inp, tgt.size(1), None, False, False)
                    mse = crit(pred, tgt).mean(dim=(2, 3, 4))
                    batch_mse = mse.mean().item()
                    mse_sum += batch_mse * mse.size(0)
                    n_seen += mse.size(0)
                    batch_pbar.set_postfix({"mse": f"{batch_mse:.4f}"})
                    if batch_pbar.n == 0:
                        log_sequence_predictions_new(inp, tgt, pred, split_name=f"vel_gen_vx{vx}_vy{vy}", num_samples=10, device=device)
            err_mat[iy, ix] = mse_sum / n_seen
            vel_pbar.update(1)
    vel_pbar.close()

    return vx_vals, vy_vals, err_mat
