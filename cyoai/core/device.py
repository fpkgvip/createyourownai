"""Device selection shared by preparation and training."""

import os

import torch


def get_device():
    override = os.environ.get("CYOAI_DEVICE")
    if override is not None:
        if override not in {"cuda", "mps", "cpu"}:
            raise ValueError("CYOAI_DEVICE must be cuda, mps, or cpu")
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
