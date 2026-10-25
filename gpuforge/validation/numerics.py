"""Numerical-correctness primitives for validating DL kernels.

Fixed ``rtol/atol`` checks are the usual way people compare kernels, but they
are wrong in both directions for reductions: too loose for small K, too tight
for large K. Here GEMM-like ops are judged against the classic forward error
bound for floating-point inner products (Higham, *Accuracy and Stability of
Numerical Algorithms*, Thm. 3.5):

    |fl(A @ B) - A @ B|  <=  gamma_K * (|A| @ |B|),   gamma_K = K*u / (1 - K*u)

where ``u`` is the unit round-off of the accumulation dtype. The reference is
computed in float64 with NumPy, independent of any GPU library, so cuBLAS,
cuDNN and custom kernels are all held to the same standard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

# unit round-off u = 2^-p (p = mantissa bits incl. implicit bit)
UNIT_ROUNDOFF = {
    torch.float64: 2.0 ** -53,
    torch.float32: 2.0 ** -24,
    torch.float16: 2.0 ** -11,
    torch.bfloat16: 2.0 ** -8,
}
# TF32 tensor cores keep 10 mantissa bits on the inputs
TF32_ROUNDOFF = 2.0 ** -11

# Element-wise ops (activation, softmax, normalisation): a few ulps of the output
ELEMENTWISE_TOL = {
    torch.float32: (1e-5, 1e-6),
    torch.float16: (2e-3, 1e-4),
    torch.bfloat16: (1.6e-2, 1e-3),
}


def gamma(n: int, u: float) -> float:
    nu = n * u
    if nu >= 1:
        return float("inf")
    return nu / (1 - nu)


def gemm_error_bound(a: np.ndarray, b: np.ndarray, u: float, safety: float = 1.0) -> np.ndarray:
    """Element-wise a-priori forward error bound for ``a @ b`` (both float64)."""
    k = a.shape[-1]
    # input quantisation adds roughly 2u more; fold it into the K term
    return safety * gamma(k + 2, u) * (np.abs(a) @ np.abs(b))


def ulp_distance_f32(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance in units-in-the-last-place between two float32 arrays."""
    def ordered(v: np.ndarray) -> np.ndarray:
        i = v.astype(np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, np.int64(-(2 ** 31)) - i, i)

    return np.abs(ordered(x) - ordered(y))


@dataclass
class CheckResult:
    op: str
    impl: str
    dtype: str
    shape: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    bound_ratio: float  # max(|err| / bound); <= 1.0 means inside the theoretical bound
    max_ulp: float | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _errors(out: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    err = np.abs(out - ref)
    rel = err / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
    return float(err.max(initial=0.0)), float(np.where(np.abs(ref) > 1e-6, rel, 0).max(initial=0.0))


def check_against_bound(op: str, impl: str, out: torch.Tensor, ref: np.ndarray,
                        bound: np.ndarray, shape: str, notes: str = "") -> CheckResult:
    o = out.detach().double().cpu().numpy()
    err = np.abs(o - ref)
    # tiny absolute floor so exact-zero bounds (e.g. zero rows) don't divide by 0
    floor = np.finfo(np.float32).tiny
    ratio = float((err / np.maximum(bound, floor)).max(initial=0.0))
    max_abs, max_rel = _errors(o, ref)
    return CheckResult(op, impl, str(out.dtype).replace("torch.", ""), shape,
                       passed=bool(np.all(err <= bound + floor)),
                       max_abs_err=max_abs, max_rel_err=max_rel, bound_ratio=ratio, notes=notes)


def check_elementwise(op: str, impl: str, out: torch.Tensor, ref: np.ndarray, shape: str,
                      rtol: float | None = None, atol: float | None = None) -> CheckResult:
    d_rtol, d_atol = ELEMENTWISE_TOL.get(out.dtype, (1e-5, 1e-6))
    rtol = d_rtol if rtol is None else rtol
    atol = d_atol if atol is None else atol
    o = out.detach().double().cpu().numpy()
    bound = atol + rtol * np.abs(ref)
    err = np.abs(o - ref)
    max_abs, max_rel = _errors(o, ref)
    ulp = None
    if out.dtype == torch.float32:
        ulp = float(ulp_distance_f32(o, ref).max(initial=0))
    return CheckResult(op, impl, str(out.dtype).replace("torch.", ""), shape,
                       passed=bool(np.all(err <= bound)), max_abs_err=max_abs,
                       max_rel_err=max_rel, bound_ratio=float((err / bound).max(initial=0.0)),
                       max_ulp=ulp)


# ------------------------------------------------------- float64 references --
def ref_gemm_bias_act(a: np.ndarray, b: np.ndarray, bias: np.ndarray | None, act: str) -> np.ndarray:
    out = a @ b
    if bias is not None:
        out = out + bias
    if act == "relu":
        out = np.maximum(out, 0)
    elif act == "gelu":
        out = 0.5 * out * (1 + np.tanh(np.sqrt(2 / np.pi) * (out + 0.044715 * out ** 3)))
    return out


def ref_conv2d(x: np.ndarray, w: np.ndarray, bias: np.ndarray | None = None,
               stride: int = 1, padding: int = 0) -> np.ndarray:
    """Direct convolution (cross-correlation, NCHW / OIHW) via im2col + einsum in float64."""
    from numpy.lib.stride_tricks import sliding_window_view

    kh, kw = w.shape[2:]
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    win = sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]
    out = np.einsum("nchwij,ocij->nohw", win, w, optimize=True)
    if bias is not None:
        out = out + bias[None, :, None, None]
    return out


def conv2d_error_bound(x: np.ndarray, w: np.ndarray, u: float, stride: int, padding: int,
                       safety: float = 1.0) -> np.ndarray:
    """Same inner-product bound as GEMM, with K = C_in * kh * kw (implicit-GEMM view)."""
    k = w.shape[1] * w.shape[2] * w.shape[3]
    return safety * gamma(k + 2, u) * ref_conv2d(np.abs(x), np.abs(w), None, stride, padding)


def ref_softmax(x: np.ndarray) -> np.ndarray:
    from scipy.special import softmax

    return softmax(x, axis=-1)
