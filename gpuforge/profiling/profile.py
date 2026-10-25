"""Profile a few training steps with torch.profiler.

    python -m gpuforge.profiling.profile --steps 10 --out reports/trace.json

Writes a Chrome/Perfetto trace and prints the top kernels by device time.
For a full system timeline use Nsight Systems - the training loop already emits
NVTX ranges per step:
    nsys profile -t cuda,nvtx,cudnn,cublas -o reports/train python -m gpuforge.train.train max_steps=50
"""
from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from gpuforge.data.pipeline import GPUPreprocessor
from gpuforge.models.resnet import build_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet_small")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--out", default="reports/trace.json")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    args = p.parse_args(argv)

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    model = build_model(args.model).to(dev).to(memory_format=torch.channels_last)
    pre = GPUPreprocessor(channels_last=True).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    x = torch.randint(0, 256, (args.batch, 3, 32, 32), dtype=torch.uint8, device=dev)
    y = torch.randint(0, 10, (args.batch,), device=dev)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=acts, schedule=schedule(wait=1, warmup=2, active=args.steps),
                 record_shapes=True, profile_memory=True) as prof:
        for _ in range(args.steps + 3):
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
                loss = torch.nn.functional.cross_entropy(model(pre(x)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            prof.step()
    prof.export_chrome_trace(args.out)
    key = "cuda_time_total" if cuda else "cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=15))
    print(f"trace written to {args.out} (open in https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
