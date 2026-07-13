#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${ROOT_DIR}/outputs/checkpoints/grpo_qwen25_3b_instruct_gsm8k_lora_2gpu26"
OUTPUT_ROOT="${ROOT_DIR}/test_model/models/grpo_checkpoints"
CONDA_ENV="${CONDA_ENV:-rl_study}"
IFS=' ' read -r -a CHECKPOINT_STEPS <<< "${STEPS:-100 200 300 400 500 600 700 800}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

for step in "${CHECKPOINT_STEPS[@]}"; do
    output_dir="${OUTPUT_ROOT}/step_${step}"
    if [[ -f "${output_dir}/lora_adapter/adapter_model.safetensors" ]]; then
        echo "step ${step}: adapter already exists"
        continue
    fi
    python test_model/export_grpo_lora.py \
        --checkpoint-dir "${CHECKPOINT_ROOT}/global_step_${step}/actor" \
        --output-dir "${output_dir}"
done
