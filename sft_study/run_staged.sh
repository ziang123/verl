#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_STEPS=5 RUN_NAME=qwen25-3b-lora-5step EVAL_STEPS=5 SAMPLE_STEPS=5 SAVE_STEPS=0 \
    "${STUDY_DIR}/run_train.sh"

MAX_STEPS=50 RUN_NAME=qwen25-3b-lora-50step EVAL_STEPS=10 SAMPLE_STEPS=10 SAVE_STEPS=50 \
    "${STUDY_DIR}/run_train.sh"

MAX_STEPS=0 RUN_NAME=qwen25-3b-lora-full EVAL_STEPS=100 SAMPLE_STEPS=100 SAVE_STEPS=250 \
    "${STUDY_DIR}/run_train.sh"
