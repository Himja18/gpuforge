"""Inference backends behind one interface: ``predict(np.ndarray[N,3,H,W]) -> logits``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Backend:
    name = "base"

    def predict(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def warmup(self, batch: int = 8, size: int = 32, iters: int = 3) -> None:
        x = np.random.randn(batch, 3, size, size).astype(np.float32)
        for _ in range(iters):
            self.predict(x)


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, ckpt: str | Path, device: str | None = None, half: bool = False):
        from gpuforge.export.export import load_checkpoint

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_checkpoint(ckpt, str(self.device))
        self.half = half and self.device.type == "cuda"
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.half:
            self.model = self.model.half()

    @torch.inference_mode()
    def predict(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x).to(self.device, non_blocking=True)
        if self.half:
            t = t.half()
        if self.device.type == "cuda":
            t = t.contiguous(memory_format=torch.channels_last)
        return self.model(t).float().cpu().numpy()


class OnnxBackend(Backend):
    name = "onnxruntime"

    def __init__(self, onnx_path: str | Path, use_gpu: bool = True):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        providers = [p for p in (["TensorrtExecutionProvider", "CUDAExecutionProvider"] if use_gpu else [])
                     if p in avail] + ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=providers)
        self.providers = self.sess.get_providers()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {"input": np.ascontiguousarray(x, dtype=np.float32)})[0]


class TensorRTBackend(Backend):
    """Runs a serialized TensorRT engine (TensorRT 10 tensor-address API).

    Device buffers are PyTorch CUDA tensors, which keeps memory management in one
    allocator and avoids a pycuda dependency.
    """

    name = "tensorrt"

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt

        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        is_half = self.engine.get_tensor_dtype(self.input) == trt.DataType.HALF
        self.in_dtype = torch.float16 if is_half else torch.float32

    @torch.inference_mode()
    def predict(self, x: np.ndarray) -> np.ndarray:
        inp = torch.from_numpy(x).to("cuda", dtype=self.in_dtype).contiguous()
        self.ctx.set_input_shape(self.input, tuple(inp.shape))
        out_shape = tuple(self.ctx.get_tensor_shape(self.output))
        out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
        self.ctx.set_tensor_address(self.input, inp.data_ptr())
        self.ctx.set_tensor_address(self.output, out.data_ptr())
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(torch.cuda.current_stream())
            if not self.ctx.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return out.cpu().numpy()


def build_backend(kind: str, artifact: str) -> Backend:
    if kind == "torch":
        return TorchBackend(artifact)
    if kind == "onnx":
        return OnnxBackend(artifact)
    if kind == "tensorrt":
        return TensorRTBackend(artifact)
    raise ValueError(f"unknown backend {kind}")
