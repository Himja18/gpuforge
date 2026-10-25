"""Python front-end for the custom CUDA kernels.

The extension is resolved in this order:
  1. a prebuilt module (``pip install -e .`` compiles it via setup.py when nvcc exists)
  2. JIT compilation with ``torch.utils.cpp_extension.load`` (``GPUFORGE_JIT=1``)
  3. a pure-PyTorch fallback, so the rest of the platform still runs on CPU-only CI.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

ACTS = {"none": 0, "relu": 1, "gelu": 2}
VARIANTS = {"naive": 0, "smem": 1, "regtile": 2}

_EXT = None
_TRIED = False


def _load_extension():
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if not torch.cuda.is_available():
        return None
    try:
        import gpuforge_kernels as ext  # type: ignore  # built by setup.py

        _EXT = ext
    except ImportError:
        if os.environ.get("GPUFORGE_JIT", "0") == "1":
            from torch.utils.cpp_extension import load

            src = Path(__file__).parent / "csrc"
            _EXT = load(
                name="gpuforge_kernels",
                sources=[str(src / "binding.cpp"), str(src / "kernels.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False,
            )
    if _EXT is None:
        log.warning("gpuforge CUDA extension not available; using PyTorch fallback")
    return _EXT


def extension_available() -> bool:
    return _load_extension() is not None


def _act_torch(x: torch.Tensor, act: str) -> torch.Tensor:
    if act == "relu":
        return F.relu(x)
    if act == "gelu":
        return F.gelu(x, approximate="tanh")
    return x


def gemm_bias_act(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None,
                  act: str = "none", variant: str = "regtile") -> torch.Tensor:
    """``act(a @ b + bias)`` with a fused epilogue on the custom kernel path."""
    if act not in ACTS:
        raise ValueError(f"act must be one of {list(ACTS)}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    ext = _load_extension()
    if ext is not None and a.is_cuda and a.dtype == torch.float32:
        return ext.gemm_bias_act(a.contiguous(), b.contiguous(),
                                 None if bias is None else bias.contiguous(),
                                 ACTS[act], VARIANTS[variant])
    out = a @ b
    if bias is not None:
        out = out + bias
    return _act_torch(out, act)


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    ext = _load_extension()
    if ext is not None and x.is_cuda and x.dtype == torch.float32 and x.dim() == 2:
        return ext.softmax_rows(x.contiguous())
    return torch.softmax(x, dim=-1)


class FusedLinear(torch.nn.Module):
    """Drop-in ``nn.Linear`` (+activation) that runs on the custom GEMM for inference."""

    def __init__(self, in_features: int, out_features: int, act: str = "none"):
        super().__init__()
        self.inner = torch.nn.Linear(in_features, out_features)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or torch.is_grad_enabled() or not x.is_cuda:
            return _act_torch(self.inner(x), self.act)  # autograd-friendly path
        w_t = self.inner.weight.t().contiguous()
        return gemm_bias_act(x.reshape(-1, x.shape[-1]), w_t, self.inner.bias, self.act) \
            .reshape(*x.shape[:-1], -1)
