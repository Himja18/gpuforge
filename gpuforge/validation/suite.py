"""Validation suite: custom kernels, cuBLAS and cuDNN vs. float64 references.

    python -m gpuforge.validation.suite --device cuda --report reports/validation.json

Exits non-zero if any check fails, so it can gate CI (see infra/jenkins/Jenkinsfile).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gpuforge import kernels
from gpuforge.utils import native_bf16
from gpuforge.validation import numerics as nm

GEMM_SHAPES = [(1, 1, 1), (7, 13, 5), (64, 64, 64), (127, 255, 129), (256, 512, 1024), (33, 1000, 4096)]
CONV_CASES = [  # (N, C, H, W, O, k, stride, pad)
    (2, 3, 32, 32, 16, 3, 1, 1),
    (4, 16, 17, 17, 32, 3, 2, 1),
    (1, 64, 14, 14, 64, 1, 1, 0),
    (2, 8, 28, 28, 8, 5, 1, 2),
]
SOFTMAX_SHAPES = [(1, 10), (64, 1000), (8, 4096), (3, 33)]


def _rng_tensor(rng: np.random.Generator, shape, scale=1.0) -> np.ndarray:
    return rng.standard_normal(shape) * scale


def _accum_roundoff(dtype: torch.dtype, device: str) -> float:
    # fp16/bf16 matmuls on CUDA accumulate in fp32, but inputs/outputs are rounded
    # to the low-precision format -> bound by the storage roundoff.
    if dtype == torch.float32 and device == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        return nm.TF32_ROUNDOFF
    return nm.UNIT_ROUNDOFF[dtype]


def _custom_gemm(a, b, bias, act, variant):
    return kernels.gemm_bias_act(a, b, bias, act, variant=variant)


def validate_gemm(device: str, rng: np.random.Generator) -> list[nm.CheckResult]:
    results = []
    impls = {"torch.matmul (cuBLAS)" if device == "cuda" else "torch.matmul (CPU BLAS)":
             lambda a, b, bias, act: kernels._act_torch(a @ b + bias, act)}
    if kernels.extension_available() and device == "cuda":
        for v in kernels.VARIANTS:
            impls[f"gpuforge.{v}"] = functools.partial(_custom_gemm, variant=v)
    dtypes = [torch.float32]
    if device == "cuda":
        dtypes.append(torch.float16)
        if native_bf16():  # skip emulated bf16 on T4-class GPUs
            dtypes.append(torch.bfloat16)

    for (m, n, k) in GEMM_SHAPES:
        a64, b64, bias64 = _rng_tensor(rng, (m, k)), _rng_tensor(rng, (k, n)), _rng_tensor(rng, (n,))
        for dtype in dtypes:
            # Reference is computed from the *quantised* inputs, so only the
            # arithmetic error of the kernel is being measured.
            a = torch.from_numpy(a64).to(dtype)
            b = torch.from_numpy(b64).to(dtype)
            bias = torch.from_numpy(bias64).to(dtype)
            aq, bq, biasq = (t.double().numpy() for t in (a, b, bias))
            u = _accum_roundoff(dtype, device)
            pre = aq @ bq + biasq  # exact pre-activation
            # error of the linear part: inner-product bound + bias add + rounding of the sum
            lin_bound = nm.gemm_error_bound(aq, bq, u) + 2 * u * np.abs(biasq) + u * np.abs(pre)
            for act in ("none", "relu", "gelu"):
                ref = nm.ref_gemm_bias_act(aq, bq, biasq, act)
                # relu is 1-Lipschitz, gelu ~1.13-Lipschitz, so input error is amplified by at
                # most that much; gelu also adds its own evaluation error (see numerics.py)
                lip = 1.13 if act == "gelu" else 1.0
                bound = lip * lin_bound + u * np.abs(ref)
                if act == "gelu":
                    bound = bound + nm.gelu_eval_error_bound(pre, u)
                for name, fn in impls.items():
                    if name.startswith("gpuforge") and dtype != torch.float32:
                        continue
                    out = fn(a.to(device), b.to(device), bias.to(device), act)
                    results.append(nm.check_against_bound(
                        f"gemm_bias_{act}", name, out, ref, bound, f"{m}x{k}@{k}x{n}"))
    return results


def validate_conv(device: str, rng: np.random.Generator) -> list[nm.CheckResult]:
    results = []
    dtypes = [torch.float32] + ([torch.float16] if device == "cuda" else [])
    impl = "F.conv2d (cuDNN)" if device == "cuda" else "F.conv2d (CPU)"
    for (n, c, h, w, o, k, s, p) in CONV_CASES:
        x64, w64, b64 = _rng_tensor(rng, (n, c, h, w)), _rng_tensor(rng, (o, c, k, k), 0.5), _rng_tensor(rng, (o,))
        for dtype in dtypes:
            x, wt, b = (torch.from_numpy(t).to(dtype) for t in (x64, w64, b64))
            xq, wq, bq = (t.double().numpy() for t in (x, wt, b))
            u = nm.TF32_ROUNDOFF if (dtype == torch.float32 and device == "cuda"
                                     and torch.backends.cudnn.allow_tf32) else nm.UNIT_ROUNDOFF[dtype]
            ref = nm.ref_conv2d(xq, wq, bq, s, p)
            # Winograd / FFT algorithms in cuDNN are not plain inner products; the
            # safety factor of 4 covers their extra transform error.
            bound = nm.conv2d_error_bound(xq, wq, u, s, p, safety=4.0) + 2 * u * np.abs(ref)
            out = F.conv2d(x.to(device), wt.to(device), b.to(device), stride=s, padding=p)
            results.append(nm.check_against_bound("conv2d", impl, out, ref, bound,
                                                  f"N{n}C{c}H{h}W{w}->O{o} k{k}s{s}p{p}"))
    return results


def validate_softmax(device: str, rng: np.random.Generator) -> list[nm.CheckResult]:
    results = []
    impls = {"torch.softmax": lambda x: torch.softmax(x, -1)}
    if kernels.extension_available() and device == "cuda":
        impls["gpuforge.softmax_rows"] = kernels.softmax_rows
    for shape in SOFTMAX_SHAPES:
        # include large logits to exercise the subtract-max stabilisation
        x64 = _rng_tensor(rng, shape, 10.0)
        x64[0, 0] = 80.0
        x = torch.from_numpy(x64).float()
        ref = nm.ref_softmax(x.double().numpy())
        for name, fn in impls.items():
            out = fn(x.to(device))
            # __expf is a fast approximation (2 ulp + range reduction) -> slightly looser
            results.append(nm.check_elementwise("softmax", name, out, ref, f"{shape[0]}x{shape[1]}",
                                                rtol=1e-5 if name == "torch.softmax" else 2e-5))
    return results


def run(device: str, seed: int = 0) -> list[nm.CheckResult]:
    rng = np.random.default_rng(seed)
    return validate_gemm(device, rng) + validate_conv(device, rng) + validate_softmax(device, rng)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report", default="reports/validation.json")
    p.add_argument("--db", default=None, help="analytics SQLite DB to record results in")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    results = run(args.device, args.seed)
    failed = [r for r in results if not r.passed]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps([r.to_dict() for r in results], indent=2))

    width = max(len(r.impl) for r in results)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"[{flag}] {r.op:<16} {r.impl:<{width}} {r.dtype:<9} {r.shape:<28} "
              f"max_abs={r.max_abs_err:.2e} bound_ratio={r.bound_ratio:.3f}")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed on {args.device}")

    if args.db:
        from gpuforge.analytics.store import AnalyticsStore

        store = AnalyticsStore(args.db)
        run_id = store.start_run("validation", {"device": args.device, "seed": args.seed})
        store.log_validations(run_id, results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
