"""Inference benchmark across backends and batch sizes (latency + throughput).

    python -m gpuforge.bench.inference --ckpt runs/latest/checkpoint.pt \
        --onnx artifacts/model.onnx --engine artifacts/model_fp16.plan --batches 1 8 32 128
"""
from __future__ import annotations

import argparse
import functools

import numpy as np

from gpuforge.bench.timing import time_fn


def bench_backend(backend, batches, iters: int = 50) -> list[dict]:
    rows = []
    on_gpu = backend.name == "tensorrt" or getattr(getattr(backend, "device", None), "type", "") == "cuda" \
        or any("CUDA" in p or "Tensorrt" in p for p in getattr(backend, "providers", []))
    device = "cuda" if on_gpu else "cpu"
    for b in batches:
        x = np.random.randn(b, 3, 32, 32).astype(np.float32)
        # predict() returns host arrays (synchronous), so wall-clock timing is end-to-end incl. H2D/D2H
        t = time_fn(functools.partial(backend.predict, x), warmup=5, iters=iters, device="cpu")
        rows.append({"name": f"infer.{backend.name}", "params": {"batch": b, "device": device},
                     "p50_ms": t["median_ms"], "p99_ms": t["p99_ms"],
                     "throughput": b / (t["median_ms"] / 1000)})
    return rows


def main(argv=None):
    from gpuforge.serve.backends import OnnxBackend, TensorRTBackend, TorchBackend

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt")
    p.add_argument("--onnx")
    p.add_argument("--engine")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--db", default="reports/analytics.db")
    args = p.parse_args(argv)

    backends = []
    if args.ckpt:
        backends.append(TorchBackend(args.ckpt))
        fp16 = TorchBackend(args.ckpt, half=True)
        if fp16.half:  # only meaningful on CUDA
            fp16.name = "torch_fp16"
            backends.append(fp16)
    if args.onnx:
        backends.append(OnnxBackend(args.onnx))
    if args.engine:
        backends.append(TensorRTBackend(args.engine))

    rows = []
    for be in backends:
        rows += bench_backend(be, args.batches, args.iters)
    print(f"{'backend':<22}{'batch':>6}{'p50 ms':>10}{'p99 ms':>10}{'img/s':>12}")
    for r in rows:
        print(f"{r['name']:<22}{r['params']['batch']:>6}{r['p50_ms']:>10.3f}{r['p99_ms']:>10.3f}{r['throughput']:>12.0f}")

    from gpuforge.analytics.store import AnalyticsStore

    store = AnalyticsStore(args.db)
    run_id = store.start_run("benchmark", {"suite": "inference", "batches": args.batches})
    for r in rows:
        store.log_benchmark(run_id, r["name"], r["params"], "p50_ms", r["p50_ms"], higher_is_better=False)
        store.log_benchmark(run_id, r["name"], r["params"], "throughput", r["throughput"], higher_is_better=True)
    print(f"recorded benchmark run #{run_id}")
    return rows


if __name__ == "__main__":
    main()
