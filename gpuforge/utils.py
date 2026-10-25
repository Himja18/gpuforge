"""Small hardware-capability helpers shared across the package."""
from __future__ import annotations

import torch


def native_bf16() -> bool:
    """True only when the GPU has hardware bf16 (Ampere / sm_80 and newer).

    Turing GPUs such as the T4 (free Colab/Kaggle tier) can *emulate* bf16, which
    is slow and not representative, so they are treated as fp16-only.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_amp_dtype(name: str, device_type: str) -> torch.dtype | None:
    """Map a config string to an autocast dtype. ``auto`` picks bf16 where it is
    native, fp16 on older GPUs, and bf16 on CPU (the only low-precision CPU autocast)."""
    if name == "none":
        return None
    if name == "auto":
        if device_type == "cuda":
            return torch.bfloat16 if native_bf16() else torch.float16
        return torch.bfloat16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"amp_dtype must be auto | bf16 | fp16 | none, got {name!r}")
