#!/usr/bin/env bash
# Start the local Qwen3.5-4B vision model as an OpenAI-compatible vLLM server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ENV="${VLLM_ENV:-${REPO_ROOT}/vllm_env}"
MODEL_PATH="${VLLM_MODEL_PATH:-/root/.cache/modelscope/models/Qwen--Qwen3.5-4B/snapshots/master}"

if [[ ! -x "${VLLM_ENV}/bin/vllm" ]]; then
    echo "vLLM is not installed in: ${VLLM_ENV}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen model config not found in: ${MODEL_PATH}" >&2
    exit 1
fi

exec "${VLLM_ENV}/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${VLLM_SERVED_MODEL_NAME:-local-qwen35-4b}" \
    --host "${VLLM_HOST:-127.0.0.1}" \
    --port "${VLLM_PORT:-8000}" \
    --dtype half \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-1}" \
    --limit-mm-per-prompt '{"image":1}'
