import numpy as np
import torch


def _unwrap(ds):
    while not hasattr(ds, "rng") and hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    np.random.seed(info.seed % (2 ** 32))
    ds = _unwrap(info.dataset)
    if getattr(ds, "random", True) is False and hasattr(ds, "rng") and hasattr(ds, "seed"):
        ds.rng = np.random.RandomState((int(ds.seed) + 9973 * (worker_id + 1)) % (2 ** 32))
