"""Data pipeline with GPU-side preprocessing.

The CPU workers only hand over raw ``uint8`` tensors (4x less PCIe traffic than
float32). Normalisation and augmentation run on the GPU as batched tensor ops,
and a side-stream prefetcher overlaps the host->device copy of batch i+1 with
compute on batch i.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class SyntheticImages(Dataset):
    """Learnable synthetic dataset: each class is a fixed random template plus noise.

    Lets the whole pipeline (and CI) run without downloading anything, while
    still giving a loss curve that should go down.
    """

    def __init__(self, n: int = 10_000, num_classes: int = 10, size: int = 32, seed: int = 0,
                 noise: float = 60.0):
        g = torch.Generator().manual_seed(seed)
        # low-frequency templates (4x4 upsampled) so class identity survives crop/flip augmentation
        coarse = torch.randint(0, 256, (num_classes, 3, 4, 4), generator=g).float()
        self.templates = torch.nn.functional.interpolate(coarse, size=(size, size), mode="bilinear",
                                                         align_corners=False)
        self.labels = torch.randint(0, num_classes, (n,), generator=g)
        self.noise_seed = seed + 1
        self.noise = noise

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        y = int(self.labels[i])
        g = torch.Generator().manual_seed(self.noise_seed * 1_000_003 + i)
        x = self.templates[y] + torch.randn(self.templates[y].shape, generator=g) * self.noise
        return x.clamp_(0, 255).to(torch.uint8), y


def build_dataset(name: str, train: bool, root: str = "data", n_synthetic: int = 10_000) -> Dataset:
    if name == "synthetic":
        return SyntheticImages(n=n_synthetic if train else max(n_synthetic // 5, 100), seed=0)
    if name == "cifar10":
        import torchvision  # optional dependency

        class _Uint8CIFAR(torchvision.datasets.CIFAR10):
            def __getitem__(self, i):
                return torch.from_numpy(self.data[i]).permute(2, 0, 1).contiguous(), int(self.targets[i])

        return _Uint8CIFAR(root, train=train, download=True)
    raise ValueError(f"unknown dataset {name}")


def build_loader(ds: Dataset, batch_size: int, train: bool, workers: int = 4,
                 distributed: bool = False) -> DataLoader:
    sampler = DistributedSampler(ds, shuffle=train) if distributed else None
    return DataLoader(ds, batch_size=batch_size, shuffle=(train and sampler is None), sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=workers > 0)


class GPUPreprocessor(torch.nn.Module):
    """uint8 NCHW -> normalised float, with batched random crop + flip on device."""

    def __init__(self, mean=CIFAR_MEAN, std=CIFAR_STD, augment: bool = True, pad: int = 4,
                 channels_last: bool = False):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)
        self.augment, self.pad, self.channels_last = augment, pad, channels_last

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() - self.mean) / self.std
        if self.augment and self.training:
            n, _, h, w = x.shape
            flip = torch.rand(n, device=x.device) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            # random crop via padding + per-sample offsets gathered with unfold-free indexing
            xp = torch.nn.functional.pad(x, [self.pad] * 4)
            dy = torch.randint(0, 2 * self.pad + 1, (n,), device=x.device)
            dx = torch.randint(0, 2 * self.pad + 1, (n,), device=x.device)
            rows = (dy.view(-1, 1) + torch.arange(h, device=x.device)).view(n, 1, h, 1)
            rows = rows.expand(-1, 3, -1, w + 2 * self.pad)
            xp = xp.gather(2, rows)
            cols = (dx.view(-1, 1) + torch.arange(w, device=x.device)).view(n, 1, 1, w).expand(-1, 3, h, -1)
            x = xp.gather(3, cols)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x


class CUDAPrefetcher:
    """Overlap H2D copies with compute using a dedicated CUDA stream."""

    def __init__(self, loader, device: torch.device):
        self.loader, self.device = loader, device
        self.cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device) if self.cuda else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.cuda:
            for x, y in self.loader:
                yield x.to(self.device), y.to(self.device)
            return
        it = iter(self.loader)
        nxt = self._preload(it)
        while nxt is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            x, y = nxt
            x.record_stream(torch.cuda.current_stream(self.device))
            y.record_stream(torch.cuda.current_stream(self.device))
            nxt = self._preload(it)
            yield x, y

    def _preload(self, it):
        try:
            x, y = next(it)
        except StopIteration:
            return None
        with torch.cuda.stream(self.stream):
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
