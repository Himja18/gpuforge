#!/bin/bash
# Multi-node DDP training on a SLURM cluster.
#   sbatch --nodes=2 infra/slurm/train_ddp.sbatch epochs=30
#SBATCH --job-name=gpuforge-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
# NCCL: InfiniBand when present, log only warnings, fail fast on hangs
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

IMAGE=${IMAGE:-ghcr.io/your-org/gpuforge:latest}

# one torchrun launcher per node; it spawns one process per GPU
srun --container-image="$IMAGE" --container-mounts="$PWD:/workspace" --container-workdir=/workspace \
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m gpuforge.train.train --config configs/train.yaml "$@"
