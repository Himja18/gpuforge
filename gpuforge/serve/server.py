"""Real-time inference server (FastAPI) with dynamic batching and Prometheus metrics.

    GPUFORGE_BACKEND=tensorrt GPUFORGE_ARTIFACT=artifacts/model_fp16.plan \
        uvicorn gpuforge.serve.server:app --host 0.0.0.0 --port 8000

Endpoints:  POST /v1/predict   GET /healthz   GET /readyz   GET /metrics
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from gpuforge.serve.batcher import DynamicBatcher

IMG_SHAPE = (3, 32, 32)


class PredictRequest(BaseModel):
    # [N, 3, 32, 32] normalised floats; flattened per-sample also accepted
    inputs: list = Field(..., description="batch of images")
    top_k: int = Field(1, ge=1, le=10)


class PredictResponse(BaseModel):
    classes: list[list[int]]
    scores: list[list[float]]


def create_app(backend=None, max_batch: int | None = None, max_wait_ms: float | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        be = backend
        if be is None:
            from gpuforge.serve.backends import build_backend

            be = build_backend(os.environ.get("GPUFORGE_BACKEND", "onnx"),
                               os.environ.get("GPUFORGE_ARTIFACT", "artifacts/model.onnx"))
        be.warmup()
        state["backend"] = be
        state["batcher"] = DynamicBatcher(
            be.predict,
            max_batch=max_batch or int(os.environ.get("GPUFORGE_MAX_BATCH", "32")),
            max_wait_ms=max_wait_ms if max_wait_ms is not None else float(os.environ.get("GPUFORGE_MAX_WAIT_MS", "2")))
        await state["batcher"].start()
        yield
        await state["batcher"].stop()

    app = FastAPI(title="GPUForge Inference", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        if "batcher" not in state:
            raise HTTPException(503, "warming up")
        return {"status": "ready", "backend": state["backend"].name}

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        try:
            x = np.asarray(req.inputs, dtype=np.float32)
            x = x.reshape(-1, *IMG_SHAPE)
        except (ValueError, TypeError) as e:
            raise HTTPException(422, f"inputs must reshape to [N, {', '.join(map(str, IMG_SHAPE))}]: {e}") from e
        if len(x) == 0 or len(x) > 256:
            raise HTTPException(422, "batch size must be between 1 and 256")
        logits = await state["batcher"].submit(x)
        z = logits - logits.max(1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        idx = np.argsort(-probs, 1)[:, :req.top_k]
        return PredictResponse(classes=idx.tolist(), scores=np.take_along_axis(probs, idx, 1).round(6).tolist())

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        s = state["batcher"].stats.snapshot() if "batcher" in state else {}
        lines = [f"# TYPE gpuforge_{k} gauge\ngpuforge_{k} {v}" for k, v in s.items()]
        return "\n".join(lines) + "\n"

    return app


app = create_app()  # backend is built lazily at startup from the env vars above
