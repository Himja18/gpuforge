"""Checkpoint -> ONNX -> TensorRT engine, with numerical parity checks at each hop.

    python -m gpuforge.export.export --ckpt runs/latest/checkpoint.pt --out artifacts/ --trt --precision fp16

TensorRT is optional: if the ``tensorrt`` Python package is missing, the exact
``trtexec`` command for building the engine is printed instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gpuforge.models.resnet import build_model


def load_checkpoint(path: str | Path, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def export_onnx(model: torch.nn.Module, path: str | Path, opset: int = 17, size: int = 32) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(2, 3, size, size)
    torch.onnx.export(model.cpu().eval(), (dummy,), str(path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
                      dynamo=False)
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))
    return path


def onnx_parity(model: torch.nn.Module, onnx_path: str | Path, batch: int = 8, size: int = 32) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(batch, 3, size, size)
    with torch.no_grad():
        ref = model.cpu().eval()(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    return {"max_abs_err": float(np.abs(out - ref).max()),
            "top1_agreement": float((out.argmax(1) == ref.argmax(1)).mean())}


def trtexec_command(onnx_path: Path, engine_path: Path, precision: str, max_batch: int, size: int) -> str:
    flags = {"fp16": "--fp16", "int8": "--int8 --fp16", "fp32": ""}[precision]
    shape = lambda b: f"input:{b}x3x{size}x{size}"
    return (f"trtexec --onnx={onnx_path} --saveEngine={engine_path} {flags} "
            f"--minShapes={shape(1)} --optShapes={shape(max(1, max_batch // 2))} --maxShapes={shape(max_batch)}")


def build_trt_engine(onnx_path: str | Path, engine_path: str | Path, precision: str = "fp16",
                     max_batch: int = 64, size: int = 32, workspace_gb: int = 2) -> Path | None:
    """Build a serialized TensorRT engine with a dynamic-batch optimisation profile."""
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    try:
        import tensorrt as trt
    except ImportError:
        print("tensorrt not installed; build the engine with:\n  "
              + trtexec_command(onnx_path, engine_path, precision, max_batch, size))
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the default in TensorRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        # Calibration-free INT8 would need Q/DQ nodes (e.g. from TensorRT Model Optimizer);
        # without them TensorRT falls back to FP16 for layers it cannot quantise.
        config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, size, size), (max(1, max_batch // 2), 3, size, size),
                      (max_batch, 3, size, size))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return engine_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--trt", action="store_true")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    p.add_argument("--max-batch", type=int, default=64)
    args = p.parse_args(argv)

    out = Path(args.out)
    model = load_checkpoint(args.ckpt)
    onnx_path = export_onnx(model, out / "model.onnx")
    parity = onnx_parity(model, onnx_path)
    print(f"ONNX exported to {onnx_path}; parity vs PyTorch: {parity}")
    if parity["top1_agreement"] < 1.0 or parity["max_abs_err"] > 1e-3:
        raise SystemExit("ONNX parity check failed")
    report = {"onnx": str(onnx_path), "onnx_parity": parity}
    if args.trt:
        eng = build_trt_engine(onnx_path, out / f"model_{args.precision}.plan", args.precision, args.max_batch)
        report["trt_engine"] = str(eng) if eng else None
    (out / "export_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
