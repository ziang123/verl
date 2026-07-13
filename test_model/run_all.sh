#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${ROOT_DIR}/test_model"
CONDA_ENV="${CONDA_ENV:-rl_study}"
RESULTS_DIR="${RESULTS_DIR:-${TEST_DIR}/results}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${RESULTS_DIR}" "${TEST_DIR}/logs"
cd "${ROOT_DIR}"

run_model() {
    local model_name="$1"
    local pids=()
    for shard in 0 1; do
        CUDA_VISIBLE_DEVICES="${shard}" python "${TEST_DIR}/evaluate_model.py" \
            --model "${model_name}" \
            --output-file "${RESULTS_DIR}/${model_name}.shard-${shard}.jsonl" \
            --shard-index "${shard}" \
            --num-shards 2 \
            --max-samples "${MAX_SAMPLES}" \
            --batch-size "${BATCH_SIZE}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            > "${TEST_DIR}/logs/${model_name}.shard-${shard}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

for model_name in base sft grpo; do
    echo "Evaluating ${model_name} on physical GPU 0 and GPU 1..."
    run_model "${model_name}"
done

if [[ "${MAX_SAMPLES}" == "0" ]]; then
    EXPECTED_COUNT=1319
else
    EXPECTED_COUNT="${MAX_SAMPLES}"
fi
python "${TEST_DIR}/summarize_results.py" --results-dir "${RESULTS_DIR}" --expected-count "${EXPECTED_COUNT}"
