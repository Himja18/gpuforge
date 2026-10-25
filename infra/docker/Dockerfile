# NGC PyTorch image ships CUDA, cuDNN, cuBLAS, NCCL, TensorRT and Nsight tooling.
ARG NGC_TAG=25.01-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG} AS build

WORKDIR /opt/gpuforge
# Compile the CUDA extension for the common data-center archs (V100, A100, L4/L40, H100)
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0" \
    PIP_NO_CACHE_DIR=1
COPY pyproject.toml setup.py ./
COPY gpuforge ./gpuforge
RUN pip install --no-build-isolation -e ".[dev]" && pip install onnxruntime-gpu

COPY configs ./configs
COPY tests ./tests

# ---- slim serving stage: same base so TensorRT/CUDA runtime libs are present
FROM build AS serve
RUN useradd -m -u 1000 gpuforge && mkdir -p /models && chown gpuforge /models
USER gpuforge
ENV GPUFORGE_BACKEND=tensorrt \
    GPUFORGE_ARTIFACT=/models/model_fp16.plan \
    GPUFORGE_MAX_BATCH=64 \
    GPUFORGE_MAX_WAIT_MS=2
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fsS localhost:8000/healthz || exit 1
CMD ["uvicorn", "gpuforge.serve.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
