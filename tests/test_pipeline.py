"""End-to-end: data -> train -> ONNX export -> serving with dynamic batching."""
import asyncio

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from gpuforge.data.pipeline import GPUPreprocessor, SyntheticImages
from gpuforge.export.export import export_onnx, load_checkpoint, onnx_parity
from gpuforge.serve.backends import OnnxBackend, TorchBackend
from gpuforge.serve.batcher import DynamicBatcher
from gpuforge.serve.server import create_app


def test_preprocessor_shapes_and_normalisation():
    pre = GPUPreprocessor(augment=True).train()
    x = torch.randint(0, 256, (16, 3, 32, 32), dtype=torch.uint8)
    out = pre(x)
    assert out.shape == (16, 3, 32, 32) and out.dtype == torch.float32
    pre.eval()
    out = pre(torch.full((2, 3, 32, 32), 255, dtype=torch.uint8))
    assert torch.allclose(out[:, 0], torch.full_like(out[:, 0], (1 - 0.4914) / 0.2470), atol=1e-4)


def test_synthetic_dataset_is_deterministic():
    ds = SyntheticImages(n=10)
    assert torch.equal(ds[3][0], ds[3][0]) and ds[3][0].dtype == torch.uint8


def test_training_logs_and_checkpoints(trained_ckpt):
    ckpt, summary = trained_ckpt
    assert ckpt.exists() and summary["steps"] > 0
    assert all(np.isfinite(h["train/loss"]) for h in summary["history"])


def test_onnx_export_parity(trained_ckpt, tmp_path):
    model = load_checkpoint(trained_ckpt[0])
    path = export_onnx(model, tmp_path / "m.onnx")
    parity = onnx_parity(model, path)
    assert parity["max_abs_err"] < 1e-4 and parity["top1_agreement"] == 1.0


def test_dynamic_batcher_coalesces_and_preserves_order():
    calls = []

    def fn(x):
        calls.append(len(x))
        return x.reshape(len(x), -1).sum(1, keepdims=True)

    async def go():
        b = DynamicBatcher(fn, max_batch=64, max_wait_ms=20)
        await b.start()
        reqs = [np.full((1, 2), i, dtype=np.float32) for i in range(20)]
        outs = await asyncio.gather(*(b.submit(r) for r in reqs))
        await b.stop()
        return outs, b.stats

    outs, stats = asyncio.run(go())
    assert [float(o[0, 0]) for o in outs] == [2.0 * i for i in range(20)]
    assert len(calls) < 20 and stats.samples == 20  # requests were merged


def test_dynamic_batcher_propagates_errors():
    def boom(x):
        raise RuntimeError("gpu fell over")

    async def go():
        b = DynamicBatcher(boom, max_wait_ms=1)
        await b.start()
        with pytest.raises(RuntimeError, match="gpu fell over"):
            await b.submit(np.zeros((1, 2), np.float32))
        await b.stop()

    asyncio.run(go())


def test_server_end_to_end(trained_ckpt, tmp_path):
    model = load_checkpoint(trained_ckpt[0])
    onnx_path = export_onnx(model, tmp_path / "m.onnx")
    backend = OnnxBackend(onnx_path, use_gpu=False)
    torch_backend = TorchBackend(trained_ckpt[0], device="cpu")
    x = np.random.randn(3, 3, 32, 32).astype(np.float32)

    with TestClient(create_app(backend, max_batch=8, max_wait_ms=1)) as client:
        assert client.get("/readyz").json()["backend"] == "onnxruntime"
        r = client.post("/v1/predict", json={"inputs": x.tolist(), "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["classes"]) == 3 and len(body["classes"][0]) == 3
        assert body["classes"][0][0] == int(torch_backend.predict(x)[0].argmax())
        assert client.post("/v1/predict", json={"inputs": [1, 2, 3]}).status_code == 422
        assert "gpuforge_requests 1" in client.get("/metrics").text
