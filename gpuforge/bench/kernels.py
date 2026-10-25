"""GEMM kernel benchmark: custom naive / smem / regtile kernels vs cuBLAS.

    python -m gpuforge.bench.kernels --sizes 512 1024 2048 4096

Reports TFLOPS (2*M*N*K / time) and % of cuBLAS, and records every result in
the analytics DB so later runs can be checked for regressions.
"""
from __future__ import annotations

import argparse
import functools

import torch

from gpuforge import kernels
from gpuforge.bench.timing import time_fn


def _custom(a, b, bias, variant):
    return kernels.gemm_bias_act(a, b, bias, "relu", variant)


def bench_gemm(sizes, device: str, iters: int = 30) -> list[dict]:
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = False  # apples-to-apples FP32 vs our FP32 kernels
    impls = {"cublas" if device == "cuda" else "cpu_blas":
             lambda a, b, bias: torch.relu(torch.addmm(bias, a, b))}
    if device == "cuda" and kernels.extension_available():
        for v in kernels.VARIANTS:
            impls[f"gpuforge.{v}"] = functools.partial(_custom, variant=v)
    if device == "cuda":
        impls["cublas_tf32"] = None  # filled below with TF32 enabled
    for n in sizes:
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
        bias = torch.randn(n, device=device)
        flops = 2 * n ** 3
        base = None
        for name, fn in impls.items():
            if name == "gpuforge.naive" and n > 2048:
                continue  # too slow to be interesting
            if name == "cublas_tf32":
                torch.backends.cuda.matmul.allow_tf32 = True
                fn = impls["cublas"]
            t = time_fn(functools.partial(fn, a, b, bias), iters=iters, device=device)
            torch.backends.cuda.matmul.allow_tf32 = False
            tflops = flops / (t["median_ms"] / 1e3) / 1e12
            if name in ("cublas", "cpu_blas"):
                base = tflops
            rows.append({"name": f"gemm.{name}", "params": {"M": n, "N": n, "K": n, "dtype": "fp32"},
                         "tflops": tflops, "median_ms": t["median_ms"],
                         "pct_of_vendor": 100 * tflops / base if base else None})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--db", default="reports/analytics.db")
    args = p.parse_args(argv)

    rows = bench_gemm(args.sizes, args.device, args.iters)
    print(f"{'kernel':<22}{'size':>7}{'median ms':>12}{'TFLOPS':>10}{'% vendor':>10}")
    for r in rows:
        pct = f"{r['pct_of_vendor']:.1f}" if r["pct_of_vendor"] is not None else "-"
        print(f"{r['name']:<22}{r['params']['M']:>7}{r['median_ms']:>12.3f}{r['tflops']:>10.3f}{pct:>10}")

    from gpuforge.analytics.store import AnalyticsStore

    store = AnalyticsStore(args.db)
    run_id = store.start_run("benchmark", {"suite": "kernels", "sizes": args.sizes})
    for r in rows:
        store.log_benchmark(run_id, r["name"], r["params"], "tflops", r["tflops"], higher_is_better=True)
    print(f"recorded benchmark run #{run_id}")
    return rows


if __name__ == "__main__":
    main()
