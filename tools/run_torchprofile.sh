#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${REPO_ROOT}/configs/gpt2.yaml}"
OUT_DIR="${2:-${REPO_ROOT}/profiles/torch}"
PROFILE_STEPS="${PROFILE_STEPS:-12}"
PROFILE_RANKS="${PROFILE_RANKS:-rank0}"

TORCH_PROF_WAIT="${TORCH_PROF_WAIT:-1}"
TORCH_PROF_WARMUP="${TORCH_PROF_WARMUP:-1}"
TORCH_PROF_ACTIVE="${TORCH_PROF_ACTIVE:-6}"
TORCH_PROF_REPEAT="${TORCH_PROF_REPEAT:-1}"

UV_BIN="${UV_BIN:-$(command -v uv)}"

if [[ -z "${UV_BIN}" ]]; then
    echo "uv not found in PATH." >&2
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

WORLD_SIZE="$(${UV_BIN} run python -c "from omegaconf import OmegaConf; c=OmegaConf.load('${CONFIG_PATH}'); print(int(c.distributed.tp_size*c.distributed.cp_size*c.distributed.pp_size*c.distributed.dp_size))")"

mkdir -p "${OUT_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MULTIGPT_ENABLE_PROFILE_RANGES=1
export MULTIGPT_ENABLE_NVTX=1

echo "[torchprofile] config=${CONFIG_PATH} world_size=${WORLD_SIZE} out=${OUT_DIR} steps=${PROFILE_STEPS}"

"${UV_BIN}" run torchrun \
    --nproc_per_node "${WORLD_SIZE}" \
    "${REPO_ROOT}/train.py" \
    --config "${CONFIG_PATH}" \
    --total_train_steps_override "${PROFILE_STEPS}" \
    --enable_torch_profiler \
    --profiler_dir "${OUT_DIR}" \
    --profiler_wait "${TORCH_PROF_WAIT}" \
    --profiler_warmup "${TORCH_PROF_WARMUP}" \
    --profiler_active "${TORCH_PROF_ACTIVE}" \
    --profiler_repeat "${TORCH_PROF_REPEAT}" \
    --profiler_profile_memory \
    --profile_ranks "${PROFILE_RANKS}"

echo "[torchprofile] traces ready under ${OUT_DIR}/rank*/"
