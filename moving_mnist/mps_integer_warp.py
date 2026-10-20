import torch
import torch.nn as nn

from velocity_model_based_MEConvLSTM_model import MEConvLSTMCell

_ORIGINAL_WARP = MEConvLSTMCell.warp
_PATCHED = False


def _integer_shift_warp(self, x, u):
    B, K, C, H, W = x.shape
    u = u.reshape(B * K, 2)
    dx = u[:, 0].round().long()
    dy = u[:, 1].round().long()

    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device), torch.arange(W, device=x.device),
        indexing="ij")

    sy = (yy.unsqueeze(0) - dy[:, None, None]) % H
    sx = (xx.unsqueeze(0) - dx[:, None, None]) % W

    idx = (sy * W + sx).reshape(B * K, 1, H * W).expand(-1, C, -1)
    out = x.reshape(B * K, C, H * W).gather(2, idx)
    return out.view(B, K, C, H, W)


def _dispatching_warp(self, x, u):
    if bool(torch.all(u == u.round())):
        return _integer_shift_warp(self, x, u)
    return _ORIGINAL_WARP(self, x, u)


def verify_equivalence(device="cpu", v_max=4, seed=0, grid_sample_atol=1e-4):
    torch.manual_seed(seed)
    cell = MEConvLSTMCell(1, 3).to(device)
    x = torch.randn(2, 3, 4, 16, 16, device=device)

    worst_roll = 0.0
    for vx in range(-v_max, v_max + 1):
        for vy in range(-v_max, v_max + 1):
            u = torch.zeros(2, 3, 2, device=device)
            u[..., 0], u[..., 1] = vx, vy
            got = _integer_shift_warp(cell, x, u)
            ref = torch.roll(x, shifts=(vy, vx), dims=(3, 4))
            worst_roll = max(worst_roll, (got - ref).abs().max().item())

    if worst_roll != 0.0:
        raise AssertionError(
            f"integer-shift warp is not an exact periodic shift: max |err| vs "
            f"torch.roll = {worst_roll:.2e}. The index map is wrong.")

    u = torch.randint(-v_max, v_max + 1, (2, 3, 2), device=device).float()
    gs_dev = (_ORIGINAL_WARP(cell, x, u) - _integer_shift_warp(cell, x, u)).abs().max().item()
    if gs_dev > grid_sample_atol:
        raise AssertionError(
            f"deviation from grid_sample is {gs_dev:.2e}, far above its ~1e-6 noise "
            f"floor -- something other than float precision differs.")
    return worst_roll, gs_dev


def enable_integer_shift_warp(force=False, verify=True, device=None):
    """Swap MEConvLSTM's warp for an exact integer-shift gather (MPS cannot backprop grid_sample)."""
    global _PATCHED
    if _PATCHED:
        return None

    needs_it = force or (device == "mps") or (
        device is None and torch.backends.mps.is_available())
    if not needs_it:
        return None

    result = verify_equivalence() if verify else None
    MEConvLSTMCell.warp = _dispatching_warp
    _PATCHED = True
    return result


def disable_integer_shift_warp():
    global _PATCHED
    MEConvLSTMCell.warp = _ORIGINAL_WARP
    _PATCHED = False
