# GPUForge — a GPU-accelerated MLOps platform

An end-to-end, production-style ML Operations stack built around NVIDIA's accelerated-computing toolchain:
**custom CUDA kernels → numerical validation against cuBLAS/cuDNN → mixed-precision multi-GPU training over NCCL →
ONNX/TensorRT real-time inference with dynamic batching → back-end performance analytics with a CI regression gate**,
deployable with Docker, Kubernetes, SLURM, Jenkins and Ansible.

```mermaid
flowchart LR
  subgraph Kernels["CUDA kernels (C++/CUDA)"]
    K1[GEMM + fused bias/act<br/>naive · smem · regtile]
    K2[Warp-shuffle softmax]
  end
  subgraph Validate["Validation framework"]
    V[float64 NumPy/SciPy reference<br/>Higham error bounds · ULP]
  end
  subgraph Train["Training"]
    D[uint8 loader → GPU preprocessing<br/>CUDA-stream prefetch] --> T[AMP bf16/fp16 · channels_last<br/>cuDNN autotune · DDP/NCCL]
  end
  subgraph Infer["Inference"]
    E[ONNX export + parity] --> R[TensorRT engine<br/>dynamic-shape profile]
    R --> S[FastAPI server<br/>dynamic batcher · /metrics]
  end
  A[(Analytics DB<br/>runs · metrics · benchmarks)]
  Kernels --> V
  cuBLAS & cuDNN --> V
  T --> E
  V --> A
  T --> A
  B[Benchmarks<br/>CUDA-event timing] --> A
  A --> G{CI regression gate<br/>Jenkins}
```

## How it maps to the NVIDIA JD (Machine Learning Operations track, plus Tools Infra / Dev Tools)

| JD requirement | Where it lives |
|---|---|
| GPU computing, accelerated computing, **CUDA** | `gpuforge/kernels/csrc/kernels.cu` — 3 GEMM variants (naive → shared-mem tiling → 64×64 block / 4×4 register tiling) with fused bias+ReLU/GELU epilogue, warp-shuffle softmax |
| **Validation frameworks for DL**, cuBLAS, cuDNN, NumPy, SciPy | `gpuforge/validation/` — validates custom kernels, `torch.matmul` (cuBLAS) and `F.conv2d` (cuDNN) in fp32/TF32/fp16/bf16 against float64 NumPy/SciPy references using *theoretical* error bounds, not magic tolerances |
| **Linear algebra & numerical methods** | Higham forward-error bound `γ_K·(|A||B|)`, ULP distance, implicit-GEMM view of convolution, stable softmax |
| **Data preprocessing** | `gpuforge/data/pipeline.py` — uint8 transfer (4× less PCIe), on-GPU normalise/crop/flip, side-stream prefetcher overlapping H2D with compute |
| **Training acceleration (CUDA, cuDNN, NCCL)** | `gpuforge/train/train.py` — AMP (bf16 / fp16 + GradScaler), TF32, `cudnn.benchmark`, channels_last, `torch.compile`, DDP over NCCL via `torchrun`, NVTX ranges for Nsight |
| **Convolution operations (cuDNN)** | ResNet backbone + cuDNN conv validation incl. strided / 1×1 / 5×5 cases |
| **Real-time inference (TensorRT)** | `gpuforge/export/` (ONNX → TensorRT FP16/INT8 engine, dynamic-batch profile, parity checks) and `gpuforge/serve/` (TensorRT 10 runtime, ONNX Runtime, PyTorch backends behind a dynamic batcher) |
| **Infrastructure for back-end analytics** | `gpuforge/analytics/` — SQLite schema for runs/metrics/benchmarks/validations + robust (median/MAD) regression detection |
| Debugging/perf tuning, application tracing | `gpuforge/profiling/` (torch.profiler → Perfetto trace) + NVTX for `nsys`; CUDA-event timing in `bench/timing.py` |
| Jenkins, Docker, Kubernetes, SLURM, Ansible, Git CI | `infra/jenkins/Jenkinsfile`, `infra/docker/Dockerfile` (NGC base), `infra/k8s/*.yaml` (`nvidia.com/gpu`), `infra/slurm/train_ddp.sbatch` (multi-node torchrun), `infra/ansible/gpu_node.yml`, `.github/workflows/ci.yml` |
| C++, Python, OOP | pybind/Torch C++ extension, typed Python package with pluggable `Backend` classes |

## Quick start

```bash
pip install --no-build-isolation -e ".[dev]"   # compiles the CUDA extension if nvcc is present
make test        # 42 tests (CUDA-kernel tests auto-skip without a GPU)
make validate    # numerical validation report -> reports/validation.json
make train       # or: make train-ddp  (all local GPUs over NCCL)
make export      # ONNX + TensorRT FP16 engine
make bench       # kernel TFLOPS vs cuBLAS, inference latency/throughput per backend
make report      # analytics summary + regression check
python -m gpuforge.analytics.report results   # measured numbers as markdown tables -> reports/RESULTS.md
make serve BACKEND=tensorrt ARTIFACT=artifacts/model_fp16.plan
```

Try the server:
```bash
python - <<'EOF'
import numpy as np, requests
x = np.random.randn(4, 3, 32, 32).tolist()
print(requests.post("http://localhost:8000/v1/predict", json={"inputs": x, "top_k": 3}).json())
EOF
curl localhost:8000/metrics     # p50/p95/p99 latency, mean batch size
```

## Run it on a free NVIDIA GPU (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Himja18/gpuforge/blob/main/notebooks/gpuforge_colab.ipynb)

Replace `Himja18` in the badge link after pushing the repo. Or open
[colab.research.google.com](https://colab.research.google.com), choose *File → Upload notebook* and pick
`notebooks/gpuforge_colab.ipynb`.

Then select *Runtime → Change runtime type → T4 GPU* and *Runtime → Run all*. In about 10–15 minutes the notebook:
builds the CUDA kernels for that GPU and checks them, runs all tests, validates kernels/cuBLAS/cuDNN, trains with
mixed precision, builds a TensorRT engine, benchmarks everything, load-tests the server's dynamic batching, profiles
training, and downloads `RESULTS.md` with your measured tables. The notebook is generated from
`notebooks/build_notebook.py`.

## Design notes (good interview material)

**Why three GEMM variants?** Each one isolates one optimisation so the benchmark explains the speed-up rather than just
reporting it. *naive* re-reads A and B from global memory K times per output. *smem* stages 32×32 tiles in shared memory,
cutting global traffic by ~32×. *regtile* has each thread compute a 4×4 output tile from registers, so every shared-memory
load feeds 4 FMAs instead of 1 — raising arithmetic intensity, which is the same idea cuBLAS/CUTLASS take much further
(tensor cores, async copies, double buffering). `ptxas -v` reports 64 registers on sm_75 (T4) and 57 on sm_80 (A100)
with 4 KB smem for *regtile*, and **zero spills** for all four kernels. Remaining gap to cuBLAS (~50 % on the T4):
no double-buffered/async global loads, no vectorised `float4` loads, and a fixed 64×64 tile for every shape.

**Why error bounds instead of `atol=1e-5`?** For an inner product of length K the rounding error grows with K. A fixed
tolerance either rejects a correct K=4096 GEMM or accepts a buggy K=16 one. The suite instead checks every element against
`γ_{K+2}·(|A||B|)` with the unit round-off of the arithmetic actually used (fp32, TF32, fp16, bf16), and reports
`bound_ratio` (how much of the error budget was used). The test suite proves the check is not vacuous: it rejects an
fp16-quality result under an fp32 bound and catches a kernel that drops a single K-slice.

**Why dynamic batching?** At batch 1 a GPU is mostly idle. The batcher waits at most `max_wait_ms` for more requests,
launches one batched inference off the event loop, and scatters results back — a bounded latency cost for a large
throughput gain. `/metrics` exposes the mean batch size so you can tune the trade-off.

**Why median/MAD for regressions?** GPU timings are noisy (clocks, thermal state, co-tenants). A benchmark is flagged
only if it is >5 % worse than the rolling median *and* outside ~3 robust standard deviations of run-to-run noise, compared
only against runs on the same device model.

## Results on an NVIDIA Tesla T4 (Google Colab, CUDA 13.0, PyTorch 2.11, TensorRT 11)

**Correctness:** all **42 tests pass** on the GPU (including 16 CUDA-kernel tests), and **all 106 numerical-validation
checks pass**. That covers the custom kernels, cuBLAS (fp32/fp16) and cuDNN (fp32/fp16) against float64 references.
The worst case used 15 % of its theoretical error budget.

### Custom GEMM kernels vs cuBLAS (FP32, median of 5 runs)

| Size (M=N=K) | naive | smem | regtile | cuBLAS | regtile % of cuBLAS | regtile vs naive |
|---|---|---|---|---|---|---|
| 1024 | 0.38 | 0.74 | **1.92** | 3.48 | 55 % | 5.1× |
| 2048 | 0.30 | 0.69 | **1.52** | 2.80 | 54 % | 5.0× |
| 4096 | — | 0.71 | **1.40** | 2.82 | 50 % | — |

TFLOPS. Each optimisation step roughly doubles throughput: shared-memory tiling ≈2× over naive, register tiling a
further 2.2–2.6×. The T4 (Turing) has no TF32 tensor cores, so cuBLAS "TF32" equals FP32 here. Run-to-run spread was
6–15 % for the custom kernels but up to 39 % for cuBLAS at 1024 on the shared Colab GPU, which is why the table uses
medians over repeated runs (a single run overstated regtile at 80 % of cuBLAS).

### Inference: ResNet backbone, 3×32×32 inputs, end-to-end including host↔device copies

| Backend | batch 1 p50 | batch 128 p50 | batch 128 throughput |
|---|---|---|---|
| PyTorch FP32 | 2.06 ms | 34.9 ms | 3,663 img/s |
| PyTorch FP16 | 2.08 ms | 4.13 ms | 31,028 img/s |
| ONNX Runtime (CUDA EP) | 0.62 ms | 26.4 ms | 4,847 img/s |
| **TensorRT FP16** | **0.55 ms** | **1.38 ms** | **93,004 img/s** |

TensorRT FP16 is **25× the throughput of PyTorch FP32** and 3× PyTorch FP16 at batch 128, with **3.7× lower batch-1
latency**. ONNX → TensorRT parity: max |Δ| 5.4e-7, 100 % top-1 agreement.

### Training

Mixed-precision (FP16 autocast + GradScaler, channels_last) ResNet training at **7,784 images/s** on one T4, loss
2.11 → 0.54 over 390 steps. (Accuracy on the synthetic dataset reaches 100 %; the dataset is a pipeline test, not a
benchmark.)

### Bugs found by running on real hardware

1. **`--use_fast_math` broke GELU accuracy.** On sm_75 it lowers `tanhf` to the `tanh.approx.f32` hardware instruction
   (34 of them in the PTX, ~2⁻¹¹ relative error). The validation suite flagged ~1 % error in the fused GEMM+GELU epilogue.
   Fix: build without fast-math, keep the explicit `__expf` intrinsic in softmax, and give GELU a documented
   evaluation-error budget (with a test that this budget still rejects the hardware approximation).
2. **TensorRT 11 is strongly typed only.** It removed the `BuilderFlag.FP16` precision switch. The exporter now converts
   the ONNX graph to fp16 (keeping fp32 I/O) and builds a strongly typed network. TensorRT 10 keeps the flag path.

## Layout

```
gpuforge/
  kernels/      CUDA C++ kernels + PyTorch binding + Python fallback
  validation/   error-bound numerics + validation suite (CI gate)
  data/         uint8 loader, GPU preprocessing, CUDA-stream prefetcher
  models/       ResNet backbone
  train/        AMP / DDP / NCCL training loop
  export/       ONNX export, TensorRT engine builder
  serve/        backends (PyTorch / ONNX Runtime / TensorRT), dynamic batcher, FastAPI app
  bench/        CUDA-event timing, kernel & inference benchmarks
  analytics/    SQLite store, regression detection, CLI reports
  profiling/    torch.profiler + NVTX
infra/          Docker, Kubernetes, SLURM, Jenkins, Ansible
configs/        training config
tests/          pytest suite
```
