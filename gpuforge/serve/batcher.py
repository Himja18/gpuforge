"""Dynamic batching: coalesce concurrent requests into one GPU launch.

A GPU is badly under-utilised at batch 1. The batcher waits at most
``max_wait_ms`` for more requests (or until ``max_batch`` samples are queued),
runs one batched forward pass off the event loop, and scatters the results back
to each caller. This trades a bounded amount of latency for much higher
throughput - the same idea as Triton Inference Server's dynamic batcher.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BatcherStats:
    requests: int = 0
    batches: int = 0
    samples: int = 0
    batch_sizes: deque = field(default_factory=lambda: deque(maxlen=10_000))
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=10_000))
    queue_ms: deque = field(default_factory=lambda: deque(maxlen=10_000))

    def snapshot(self) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.zeros(1)
        return {
            "requests": self.requests, "batches": self.batches, "samples": self.samples,
            "mean_batch_size": float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0,
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "latency_p99_ms": float(np.percentile(lat, 99)),
            "queue_p50_ms": float(np.percentile(self.queue_ms, 50)) if self.queue_ms else 0.0,
        }


class DynamicBatcher:
    def __init__(self, predict_fn, max_batch: int = 32, max_wait_ms: float = 2.0):
        self.predict_fn = predict_fn
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stats = BatcherStats()
        self._task: asyncio.Task | None = None

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, x: np.ndarray) -> np.ndarray:
        """``x``: [n, ...] samples from one request; returns their outputs."""
        fut = asyncio.get_running_loop().create_future()
        t0 = time.perf_counter()
        await self.queue.put((x, fut, t0))
        out = await fut
        self.stats.requests += 1
        self.stats.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return out

    async def _loop(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            n = len(items[0][0])
            deadline = loop.time() + self.max_wait
            while n < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout)
                except asyncio.TimeoutError:
                    break
                items.append(item)
                n += len(item[0])
            now = time.perf_counter()
            for _, _, t0 in items:
                self.stats.queue_ms.append((now - t0) * 1000)
            batch = np.concatenate([it[0] for it in items], axis=0)
            try:
                # run the blocking forward pass in a worker thread so the event loop keeps accepting requests
                out = await loop.run_in_executor(None, self.predict_fn, batch)
            except Exception as e:  # propagate to every waiting caller
                for _, fut, _ in items:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            self.stats.batches += 1
            self.stats.samples += len(batch)
            self.stats.batch_sizes.append(len(batch))
            i = 0
            for x, fut, _ in items:
                if not fut.done():
                    fut.set_result(out[i:i + len(x)])
                i += len(x)
