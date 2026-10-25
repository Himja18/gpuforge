.PHONY: install build test validate train export bench serve profile report docker clean

install:            ## install package (+ builds CUDA extension if nvcc is present)
	pip install --no-build-isolation -e ".[dev]"

build:              ## (re)compile only the CUDA extension
	python setup.py build_ext --inplace

test:
	pytest -q

validate:           ## numerical validation: custom kernels / cuBLAS / cuDNN vs float64
	python -m gpuforge.validation.suite --db reports/analytics.db

train:
	python -m gpuforge.train.train --config configs/train.yaml

train-ddp:          ## all local GPUs over NCCL
	torchrun --standalone --nproc_per_node=$$(nvidia-smi -L | wc -l) -m gpuforge.train.train --config configs/train.yaml

export:
	python -m gpuforge.export.export --ckpt runs/latest/checkpoint.pt --out artifacts --trt --precision fp16

bench:
	python -m gpuforge.bench.kernels
	python -m gpuforge.bench.inference --ckpt runs/latest/checkpoint.pt --onnx artifacts/model.onnx \
		$$(test -f artifacts/model_fp16.plan && echo --engine artifacts/model_fp16.plan)

serve:
	GPUFORGE_BACKEND=$${BACKEND:-onnx} GPUFORGE_ARTIFACT=$${ARTIFACT:-artifacts/model.onnx} \
		uvicorn gpuforge.serve.server:app --host 0.0.0.0 --port 8000

profile:
	python -m gpuforge.profiling.profile --out reports/trace.json

report:
	python -m gpuforge.analytics.report summary
	python -m gpuforge.analytics.report regress

all: validate train export bench report   ## full pipeline on a GPU box

docker:
	docker build -f infra/docker/Dockerfile -t gpuforge:latest .

clean:
	rm -rf build *.egg-info runs artifacts reports **/__pycache__ gpuforge_kernels*.so
