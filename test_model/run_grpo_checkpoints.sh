#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${ROOT_DIR}/test_model"
RESULTS_DIR="${RESULTS_DIR:-${TEST_DIR}/results/grpo_checkpoint_comparison}"
ADAPTER_ROOT="${TEST_DIR}/models/grpo_checkpoints"
CONDA_ENV="${CONDA_ENV:-rl_study}"
SAMPLE_COUNT="${SAMPLE_COUNT:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IFS=' ' read -r -a CHECKPOINT_STEPS <<< "${STEPS:-100 200 300 400 500 600 700 800}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${RESULTS_DIR}" "${TEST_DIR}/logs"
cd "${ROOT_DIR}"

for step in "${CHECKPOINT_STEPS[@]}"; do
    echo "Evaluating GRPO step ${step}..."
    pids=()
    for shard in 0 1; do
        CUDA_VISIBLE_DEVICES="${shard}" python test_model/evaluate_model.py \
            --model grpo \
            --model-name "grpo_step_${step}" \
            --grpo-adapter-path "${ADAPTER_ROOT}/step_${step}/lora_adapter" \
            --output-file "${RESULTS_DIR}/grpo_step_${step}.shard-${shard}.jsonl" \
            --sample-count "${SAMPLE_COUNT}" \
            --shard-index "${shard}" \
            --num-shards 2 \
            --batch-size "${BATCH_SIZE}" \
            > "${TEST_DIR}/logs/grpo_step_${step}.shard-${shard}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
done

python test_model/compare_grpo_checkpoints.py \
    --results-dir "${RESULTS_DIR}" \
    --expected-count "${SAMPLE_COUNT}" \
    --training-summary "${RESULTS_DIR}/training_rollout_summary.csv" \
    --steps "${CHECKPOINT_STEPS[@]}"
