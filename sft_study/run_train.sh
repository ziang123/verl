#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY_DIR="${ROOT_DIR}/sft_study"
CONDA_ENV="${CONDA_ENV:-rl_study}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${STUDY_DIR}/.matplotlib}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${CUDA_VISIBLE_DEVICES}" != "0,1" ]]; then
    echo "ERROR: this study is pinned to CUDA_VISIBLE_DEVICES=0,1, got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

MODEL_PATH="${MODEL_PATH:-/media/iie/4Tb/model/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/sft-data}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
MAX_STEPS="${MAX_STEPS:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
RUN_NAME="${RUN_NAME:-qwen25-3b-lora-$([[ ${MAX_STEPS} == 0 ]] && echo full || echo ${MAX_STEPS}step)}"
OUTPUT_DIR="${OUTPUT_DIR:-${STUDY_DIR}/outputs/${RUN_NAME}}"
LOG_DIR="${STUDY_DIR}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${MPLCONFIGDIR}"
cd "${ROOT_DIR}"

if [[ ! -f "${DATA_DIR}/train.jsonl" || ! -f "${DATA_DIR}/test.jsonl" ]]; then
    python "${STUDY_DIR}/preprocess_data.py"
fi

CMD=(
    accelerate launch
    --multi_gpu
    --num_processes 2
    --num_machines 1
    --gpu_ids 0,1
    --mixed_precision "${MIXED_PRECISION}"
    --dynamo_backend no
    "${STUDY_DIR}/train_lora.py"
    --model-path "${MODEL_PATH}"
    --train-file "${DATA_DIR}/train.jsonl"
    --eval-file "${DATA_DIR}/test.jsonl"
    --output-dir "${OUTPUT_DIR}"
    --mixed-precision "${MIXED_PRECISION}"
    --max-steps "${MAX_STEPS}"
    --num-train-epochs "${NUM_TRAIN_EPOCHS}"
    --max-length "${MAX_LENGTH:-768}"
    --max-eval-samples "${MAX_EVAL_SAMPLES:-128}"
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
    --learning-rate "${LEARNING_RATE:-1e-4}"
    --logging-steps "${LOGGING_STEPS:-1}"
    --eval-steps "${EVAL_STEPS:-10}"
    --sample-steps "${SAMPLE_STEPS:-10}"
    --save-steps "${SAVE_STEPS:-50}"
    --num-generation-samples "${NUM_GENERATION_SAMPLES:-1}"
    --generation-max-new-tokens "${GENERATION_MAX_NEW_TOKENS:-192}"
)

printf 'Run directory: %s\nLog file: %s\nCommand: ' "${OUTPUT_DIR}" "${LOG_FILE}"
printf '%q ' "${CMD[@]}"
printf '\n'

set +e
{
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ${RUN_NAME} ====="
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
    "${CMD[@]}" "$@"
} 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e
exit "${STATUS}"
