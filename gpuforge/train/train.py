"""Accelerated training: AMP, channels_last, cuDNN autotuning, torch.compile and
multi-GPU DDP over NCCL.

Single GPU:   python -m gpuforge.train.train --config configs/train.yaml
Multi GPU:    torchrun --nproc_per_node=8 -m gpuforge.train.train --config configs/train.yaml
SLURM:        sbatch infra/slurm/train_ddp.sbatch
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from gpuforge.data.pipeline import CUDAPrefetcher, GPUPreprocessor, build_dataset, build_loader
from gpuforge.models.resnet import build_model


@dataclass
class TrainConfig:
    model: str = "resnet_small"
    dataset: str = "synthetic"
    n_synthetic: int = 10_000
    epochs: int = 5
    batch_size: int = 256          # per GPU
    lr: float = 0.1                # peak LR for batch 256; scaled linearly with world size
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    amp_dtype: str = "bf16"        # bf16 | fp16 | none
    channels_last: bool = True
    compile: bool = False
    workers: int = 4
    max_steps: int = 0             # >0 stops early (smoke tests / benchmarking)
    log_every: int = 20
    out_dir: str = "runs/latest"
    db: str = "reports/analytics.db"
    seed: int = 0

    @classmethod
    def load(cls, path: str | None, overrides: list[str]) -> TrainConfig:
        data = {}
        if path:
            import yaml

            data = yaml.safe_load(Path(path).read_text()) or {}
        types = {f.name: f.type for f in fields(cls)}
        for kv in overrides:
            k, v = kv.split("=", 1)
            if k not in types:
                raise KeyError(f"unknown config key {k}")
            data[k] = yaml_scalar(v)
        return cls(**data)


def yaml_scalar(v: str):
    import yaml

    return yaml.safe_load(v)


# ------------------------------------------------------------ distributed ---
def setup_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, local, device


def nvtx(name: str):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)  # shows up in Nsight Systems timelines
    return contextlib.nullcontext()


def _amp(cfg: TrainConfig, device: torch.device):
    if cfg.amp_dtype == "none":
        return contextlib.nullcontext, None
    dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(device.type) if (dtype == torch.float16 and device.type == "cuda") else None
    return (lambda: torch.autocast(device.type, dtype=dtype)), scaler


@torch.no_grad()
def evaluate(model: nn.Module, loader, pre: GPUPreprocessor, device, autocast) -> float:
    model.eval()
    pre.eval()
    correct = total = 0
    for x, y in CUDAPrefetcher(loader, device):
        with autocast():
            logits = model(pre(x))
        correct += (logits.argmax(1) == y).sum()
        total += y.numel()
    t = torch.tensor([float(correct), float(total)], device=device)
    if dist.is_initialized():
        dist.all_reduce(t)
    model.train()
    pre.train()
    return float(t[0] / t[1])


def train(cfg: TrainConfig) -> dict:
    rank, world, local, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)
    is_main = rank == 0
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True          # autotune conv algorithms per shape
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 tensor cores on Ampere+
        torch.backends.cudnn.allow_tf32 = True

    train_ds = build_dataset(cfg.dataset, True, n_synthetic=cfg.n_synthetic)
    val_ds = build_dataset(cfg.dataset, False, n_synthetic=cfg.n_synthetic)
    train_loader = build_loader(train_ds, cfg.batch_size, True, cfg.workers, world > 1)
    val_loader = build_loader(val_ds, cfg.batch_size * 2, False, cfg.workers, world > 1)

    model = build_model(cfg.model).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    pre = GPUPreprocessor(channels_last=cfg.channels_last).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None,
                    gradient_as_bucket_view=True)
    fwd = torch.compile(model) if cfg.compile else model

    # no weight decay on BN / bias parameters
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    lr = cfg.lr * cfg.batch_size * world / 256
    opt = torch.optim.SGD([{"params": decay, "weight_decay": cfg.weight_decay},
                           {"params": no_decay, "weight_decay": 0.0}], lr=lr, momentum=0.9, nesterov=True)
    steps_per_epoch = len(train_loader)
    total_steps = cfg.max_steps or cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    autocast, scaler = _amp(cfg, device)

    store = run_id = None
    if is_main:
        from gpuforge.analytics.store import AnalyticsStore

        store = AnalyticsStore(cfg.db)
        run_id = store.start_run("train", {**asdict(cfg), "world_size": world})
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    step, t_last, imgs, history = 0, time.perf_counter(), 0, []
    done = False
    for epoch in range(cfg.epochs if not cfg.max_steps else 10 ** 9):
        if world > 1:
            train_loader.sampler.set_epoch(epoch)
        for x, y in CUDAPrefetcher(train_loader, device):
            with nvtx(f"step{step}"):
                with autocast():
                    loss = loss_fn(fwd(pre(x)), y)
                opt.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                sched.step()
            step += 1
            imgs += x.shape[0] * world
            if step % cfg.log_every == 0 or step == total_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t_last
                rec = {"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0],
                       "train/images_per_sec": imgs / dt}
                if device.type == "cuda":
                    rec["train/max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                if is_main:
                    store.log_metrics(run_id, step, rec)
                    vals = " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in rec.items())
                    print(f"step {step}/{total_steps} {vals}", flush=True)
                history.append(rec)
                t_last, imgs = time.perf_counter(), 0
            if step >= total_steps:
                done = True
                break
        acc = evaluate(model, val_loader, pre, device, autocast)
        if is_main:
            store.log_metrics(run_id, step, {"val/accuracy": acc})
            print(f"epoch {epoch} val_acc={acc:.4f}", flush=True)
            raw = model.module if isinstance(model, DDP) else model
            torch.save({"model": raw.state_dict(), "config": asdict(cfg), "step": step, "val_acc": acc},
                       Path(cfg.out_dir) / "checkpoint.pt")
        if done:
            break

    summary = {"steps": step, "val_acc": acc, "run_id": run_id, "history": history}
    if is_main:
        (Path(cfg.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. epochs=1 amp_dtype=fp16")
    args = p.parse_args(argv)
    train(TrainConfig.load(args.config, args.overrides))


if __name__ == "__main__":
    main()
