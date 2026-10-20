import torch


from itertools import permutations as _permutations

import torch
import numpy as np


class VelocityMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct    = None
        self.total      = None
        self.l2_sum     = None
        self.correct_sw = None
        self.l2_sum_sw  = None
        self.examples   = {}
        self._T         = None

    @staticmethod
    def _perm_matches(pred_vel, true_vel):
        B, T, K, _ = pred_vel.shape
        N          = true_vel.shape[2]
        assert K >= N, f"n_slots ({K}) must be >= num_digits ({N})"

        perms  = torch.tensor(list(_permutations(range(K), N)), dtype=torch.long)
        pred_r = torch.round(pred_vel)

        match = torch.stack([
            (pred_r[:, :, p, :] == true_vel).all(dim=-1)
            for p in perms.tolist()
        ], dim=0)
        return match, perms

    @staticmethod
    def _gather_perm(pred_vel, sel):
        B, T, N = sel.shape
        return torch.gather(pred_vel, 2, sel.unsqueeze(-1).expand(B, T, N, 2))

    @classmethod
    def _best_assignment(cls, pred_vel, true_vel):
        match, perms = cls._perm_matches(pred_vel, true_vel)
        B, T = pred_vel.shape[:2]
        N    = true_vel.shape[2]
        best = match.sum(dim=(2, 3)).argmax(dim=0)
        sel  = perms[best]
        matched = cls._gather_perm(pred_vel, sel.unsqueeze(1).expand(B, T, N))
        return matched, sel

    def update(self, pred_vel, true_vel):
        pred_vel = pred_vel.detach().float().cpu()
        true_vel = true_vel.detach().float().cpu()

        T = min(pred_vel.size(1), true_vel.size(1))
        pred_vel = pred_vel[:, :T]
        true_vel = true_vel[:, :T]
        B, T, K, _ = pred_vel.shape

        N = true_vel.shape[2]

        if self._T is None:
            self._T         = T
            self.correct    = torch.zeros(T, dtype=torch.long)
            self.total      = torch.zeros(T, dtype=torch.long)
            self.l2_sum     = torch.zeros(T)
            self.correct_sw = torch.zeros(T, dtype=torch.long)
            self.l2_sum_sw  = torch.zeros(T)

        match, perms = self._perm_matches(pred_vel, true_vel)
        bx = torch.arange(B)

        seq_best   = match.sum(dim=(2, 3)).argmax(dim=0)
        seq_sel    = perms[seq_best].unsqueeze(1).expand(B, T, N)
        matched    = self._gather_perm(pred_vel, seq_sel)
        match_seq  = match[seq_best, bx]
        l2_seq     = torch.norm(matched - true_vel, dim=-1)

        step_best  = match.sum(dim=3).argmax(dim=0)
        step_sel   = perms[step_best]
        matched_sw = self._gather_perm(pred_vel, step_sel)
        match_sw   = match[step_best, bx[:, None], torch.arange(T)[None, :]]
        l2_sw      = torch.norm(matched_sw - true_vel, dim=-1)

        for t in range(T):
            self.correct[t]    += match_seq[:, t].sum().item()
            self.total[t]      += match_seq[:, t].numel()
            self.l2_sum[t]     += l2_seq[:, t].sum().item()
            self.correct_sw[t] += match_sw[:, t].sum().item()
            self.l2_sum_sw[t]  += l2_sw[:, t].sum().item()

            if t not in self.examples:
                bad = (~match_sw[:, t]).nonzero(as_tuple=False)
                if len(bad) > 0:
                    b, n = bad[0, 0].item(), bad[0, 1].item()
                    self.examples[t] = {
                        "batch"       : b,
                        "digit"       : n,
                        "step_perm"   : step_sel[b, t].tolist(),
                        "seq_perm"    : seq_sel[b, t].tolist(),
                        "pred_float"  : tuple(round(x, 2) for x in
                                              matched_sw[b, t, n].tolist()),
                        "pred_round"  : tuple(int(x) for x in
                                              torch.round(matched_sw[b, t, n]).tolist()),
                        "true"        : tuple(int(x) for x in
                                              true_vel[b, t, n].tolist()),
                    }

    def summary(self):
        if self._T is None:
            return None

        per_t_total = self.total.clamp(min=1)
        per_t_acc   = (100.0 * self.correct.float() / per_t_total).tolist()
        per_t_l2    = (self.l2_sum / per_t_total).tolist()

        overall_total = max(1, self.total.sum().item())
        overall_acc   = 100.0 * self.correct.sum().item() / overall_total
        overall_l2    = self.l2_sum.sum().item() / overall_total

        half      = self._T // 2
        enc_total = max(1, self.total[:half].sum().item())
        dec_total = max(1, self.total[half:].sum().item())
        encoder_acc = 100.0 * self.correct[:half].sum().item() / enc_total
        decoder_acc = 100.0 * self.correct[half:].sum().item() / dec_total

        per_t_acc_sw = (100.0 * self.correct_sw.float() / per_t_total).tolist()
        per_t_l2_sw  = (self.l2_sum_sw / per_t_total).tolist()
        overall_acc_sw = 100.0 * self.correct_sw.sum().item() / overall_total
        overall_l2_sw  = self.l2_sum_sw.sum().item() / overall_total
        encoder_acc_sw = 100.0 * self.correct_sw[:half].sum().item() / enc_total
        decoder_acc_sw = 100.0 * self.correct_sw[half:].sum().item() / dec_total

        per_t_binding_loss = [a - b for a, b in zip(per_t_acc_sw, per_t_acc)]

        return {
            "T"             : self._T,
            "per_t_acc"     : per_t_acc,
            "per_t_l2"      : per_t_l2,
            "per_t_correct" : self.correct.tolist(),
            "per_t_total"   : self.total.tolist(),
            "overall_acc"   : overall_acc,
            "overall_l2"    : overall_l2,
            "encoder_acc"   : encoder_acc,
            "decoder_acc"   : decoder_acc,
            "last_step_acc" : per_t_acc[-1],
            "last_step_l2"  : per_t_l2[-1],
            "per_t_acc_stepwise"     : per_t_acc_sw,
            "per_t_l2_stepwise"      : per_t_l2_sw,
            "per_t_correct_stepwise" : self.correct_sw.tolist(),
            "overall_acc_stepwise"   : overall_acc_sw,
            "overall_l2_stepwise"    : overall_l2_sw,
            "encoder_acc_stepwise"   : encoder_acc_sw,
            "decoder_acc_stepwise"   : decoder_acc_sw,
            "last_step_acc_stepwise" : per_t_acc_sw[-1],
            "last_step_l2_stepwise"  : per_t_l2_sw[-1],
            "per_t_binding_loss"     : per_t_binding_loss,
        }

    def report(self, title="Velocity Report"):
        s = self.summary()
        if s is None:
            print("No data recorded yet.")
            return

        print()
        print("=" * 100)
        print(title)
        print(
            "  acc%      : slot→digit assignment chosen AT EACH STEP — how good the\n"
            "              velocity estimate at time t is, on its own. Read this one.\n"
            "  acc%(seq) : one assignment per sequence, chosen across all timesteps.\n"
            "              Lower whenever a step's own binding disagrees with the rest\n"
            "              of the sequence; it mixes estimate quality with binding\n"
            "              consistency, so it can move even for a parameter-free step.\n"
            "  bind      : acc% - acc%(seq), i.e. how much is lost purely to the\n"
            "              binding disagreeing across time (never negative).\n"
            "  t=2 low is structural: both slots have identical hidden states after\n"
            "  t=1 (warp(0,v)=0), so they report one velocity and can match at most\n"
            "  one of the N digits."
        )
        print("=" * 100)
        print(f"{'t':>4}  {'acc%':>8}  {'acc%(seq)':>10}  {'bind':>6}  "
              f"{'mean_L2':>8}  {'L2(seq)':>8}  {'correct/total':>15}  note")
        print("-" * 100)

        for t in range(s["T"]):
            note = ""
            if t == 1:
                note = "← slot degeneracy (both slots identical at this step)"

            row = (f"{t+1:>4}  {s['per_t_acc_stepwise'][t]:>8.2f}  "
                   f"{s['per_t_acc'][t]:>10.2f}  "
                   f"{s['per_t_binding_loss'][t]:>6.2f}  "
                   f"{s['per_t_l2_stepwise'][t]:>8.3f}  {s['per_t_l2'][t]:>8.3f}"
                   f"  {s['per_t_correct_stepwise'][t]:>7}/{s['per_t_total'][t]:<7}  {note}")
            print(row)

            if t in self.examples:
                e = self.examples[t]
                print(f"       first error:  digit={e['digit']}"
                      f"  step_perm={e['step_perm']}"
                      f"  seq_perm={e['seq_perm']}"
                      f"  pred_float={e['pred_float']}"
                      f"  pred_round={e['pred_round']}"
                      f"  true={e['true']}")

        print("-" * 100)
        print(f"Overall accuracy (per-step assignment) : {s['overall_acc_stepwise']:.2f}%")
        print(f"Overall accuracy (sequence-locked)     : {s['overall_acc']:.2f}%")
        print(f"Overall mean L2  (per-step assignment) : {s['overall_l2_stepwise']:.3f}")
        print(f"Overall mean L2  (sequence-locked)     : {s['overall_l2']:.3f}")
        print("=" * 100)
        print()

        half = s["T"] // 2
        print(f"  Encoder steps (t=1..{half})   accuracy: "
              f"{s['encoder_acc_stepwise']:.2f}%  (seq-locked {s['encoder_acc']:.2f}%)")
        print(f"  Decoder steps (t={half+1}..{s['T']}) accuracy: "
              f"{s['decoder_acc_stepwise']:.2f}%  (seq-locked {s['decoder_acc']:.2f}%)")
        print()


def motions_to_true_vel(motions, T_estimated):
    T = min(T_estimated, motions.shape[1])
    return motions[:, :T].float()
