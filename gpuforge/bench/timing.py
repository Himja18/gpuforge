"""Correct GPU timing. Kernel launches are asynchronous, so wall-clock timing
around a launch measures the launch, not the kernel. On CUDA we time with
events on the stream, after warm-up, and report robust statistics."""
from __future__ import annotations

import statistics
import time

import torch


def time_fn(fn, warmup: int = 5, iters: int = 50, device: str = "cuda") -> dict:
    for _ in range(warmup):
        fn()
    times = []
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for s, e in zip(starts, ends, strict=True):
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    pct = lambda q: times[min(len(times) - 1, int(q * len(times)))]
    return {"mean_ms": statistics.fmean(times), "median_ms": statistics.median(times),
            "p95_ms": pct(0.95), "p99_ms": pct(0.99), "min_ms": times[0]}
