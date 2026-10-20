import math

import torch


def bias_for_tau(tau):
    lam = math.exp(-1.0 / float(tau))
    return math.log(lam / (1.0 - lam))


@torch.no_grad()
def set_forget_bias(conv, hidden_channels, short=None, long=None, long_fraction=0.5):
    if conv.bias is None or short is None:
        return
    f = conv.bias[hidden_channels:2 * hidden_channels]
    f.fill_(float(short))
    if long is not None:
        n_long = int(round(long_fraction * hidden_channels))
        f[:n_long] = float(long)
