#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${REPO_ROOT}/configs/gpt2.yaml}"
OUT_DIR="${2:-${REPO_ROOT}/profiles/nsys}"
RUN_NAME="${RUN_NAME:-tp_overlap_$(date +%Y%m%d_%H%M%S)}"

NSYS_TOTAL_STEPS="${NSYS_TOTAL_STEPS:-12}"
NSYS_START_STEP="${NSYS_START_STEP:-2}"
NSYS_STOP_STEP="${NSYS_STOP_STEP:-8}"

UV_BIN="${UV_BIN:-$(command -v uv)}"

if [[ -z "${UV_BIN}" ]]; then
    echo "uv not found in PATH." >&2
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found in PATH. Please install Nsight Systems first." >&2
    exit 1
fi

WORLD_SIZE="$(${UV_BIN} run python -c "from omegaconf import OmegaConf; c=OmegaConf.load('${CONFIG_PATH}'); print(int(c.distributed.tp_size*c.distributed.cp_size*c.distributed.pp_size*c.distributed.dp_size))")"

mkdir -p "${OUT_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MULTIGPT_ENABLE_NVTX=1
export MULTIGPT_ENABLE_PROFILE_RANGES=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "[nsys] config=${CONFIG_PATH} world_size=${WORLD_SIZE} out=${OUT_DIR}/${RUN_NAME}"

nsys profile \
    --output "${OUT_DIR}/${RUN_NAME}" \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace-fork-before-exec=true \
    --wait=all \
    "${UV_BIN}" run torchrun \
    --nproc_per_node "${WORLD_SIZE}" \
    "${REPO_ROOT}/train.py" \
    --config "${CONFIG_PATH}" \
    --total_train_steps_override "${NSYS_TOTAL_STEPS}" \
    --enable_cuda_profiler_api \
    --cuda_profiler_start_step "${NSYS_START_STEP}" \
    --cuda_profiler_stop_step "${NSYS_STOP_STEP}"

echo "[nsys] report files: ${OUT_DIR}/${RUN_NAME}.nsys-rep and .sqlite"
