#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-rl_study}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$("${CONDA_EXE}" info --base)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    else
        echo "ERROR: conda is not available. Activate ${CONDA_ENV} or set PYTHON_BIN." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,6}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${CUDA_VISIBLE_DEVICES}" != "2,6" ]]; then
    echo "ERROR: this test is pinned to CUDA_VISIBLE_DEVICES=2,6, got ${CUDA_VISIBLE_DEVICES}." >&2
    exit 2
fi

MODEL_PATH="${MODEL_PATH:-/media/iie/4Tb/model/Qwen2.5-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT_DIR}/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-${ROOT_DIR}/data/gsm8k/test.parquet}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-4}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:--1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
ACTOR_LR="${ACTOR_LR:-3e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
PROJECT_NAME="${PROJECT_NAME:-verl_qwen25_3b_instruct_lora}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen25_3b_instruct_gsm8k_lora_2gpu26}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${LOG_DIR}/rollouts/${EXPERIMENT_NAME}}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/outputs/checkpoints/${EXPERIMENT_NAME}}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/verl_grpo_ray}"
LIVE_PLOT="${LIVE_PLOT:-1}"
PLOT_INTERVAL="${PLOT_INTERVAL:-30}"
RUN_TRAINING="${RUN_TRAINING:-1}"

if [[ "${NGPUS_PER_NODE}" != "2" ]]; then
    echo "ERROR: this test requires trainer.n_gpus_per_node=2, got ${NGPUS_PER_NODE}." >&2
    exit 2
fi

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}" "${ROLLOUT_DATA_DIR}" "${CKPT_DIR}" "${RAY_TMPDIR}"

check_backend() {
    case "${ROLLOUT_BACKEND}" in
        vllm)
            "${PYTHON_BIN}" - <<'PY'
import vllm
print(f"vllm={vllm.__version__}")
PY
            ;;
        sglang)
            "${PYTHON_BIN}" - <<'PY'
import sglang
print(f"sglang={sglang.__version__}")
PY
            ;;
        *)
            echo "ERROR: ROLLOUT_BACKEND must be vllm or sglang for current verl main, got ${ROLLOUT_BACKEND}." >&2
            exit 3
            ;;
    esac
}

check_inputs() {
    [[ -f "${TRAIN_FILE}" ]] || { echo "ERROR: missing TRAIN_FILE=${TRAIN_FILE}" >&2; exit 4; }
    [[ -f "${VAL_FILE}" ]] || { echo "ERROR: missing VAL_FILE=${VAL_FILE}" >&2; exit 4; }
    [[ -d "${MODEL_PATH}" ]] || { echo "ERROR: missing MODEL_PATH=${MODEL_PATH}" >&2; exit 4; }
    "${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, visible_devices={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("ERROR: need two visible CUDA devices")
PY
    check_backend
}

CMD=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    +ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
    +ray_kwargs.ray_init.num_gpus="${NGPUS_PER_NODE}"
    +ray_kwargs.ray_init.include_dashboard=False
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}"
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.dataloader_num_workers=0
    actor_rollout_ref.model.path="${MODEL_PATH}"
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank="${LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    reward.num_workers="${REWARD_NUM_WORKERS}"
    trainer.balance_batch=True
    trainer.use_v1=False
    trainer.logger='["console","tensorboard"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.resume_mode=disable
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    CMD+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

check_inputs

printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${RUN_TRAINING}" == "0" ]]; then
    echo "RUN_TRAINING=0, configuration check finished."
    exit 0
fi

: > "${LOG_FILE}"
PLOT_PID=""
cleanup_live_plot() {
    if [[ -n "${PLOT_PID}" ]] && kill -0 "${PLOT_PID}" >/dev/null 2>&1; then
        kill "${PLOT_PID}" >/dev/null 2>&1 || true
        wait "${PLOT_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup_live_plot EXIT

if [[ "${LIVE_PLOT}" != "0" ]]; then
    "${PYTHON_BIN}" scripts/plot_training_metrics.py "${LOG_FILE}" --watch --interval "${PLOT_INTERVAL}" \
        > "${LOG_FILE%.log}.plot.log" 2>&1 &
    PLOT_PID="$!"
    echo "Live loss plot: ${LOG_FILE%.log}.loss.png"
fi

set +e
{
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ${EXPERIMENT_NAME} ====="
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "TRAIN_FILE=${TRAIN_FILE}"
    echo "VAL_FILE=${VAL_FILE}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR}"
    echo "RAY_NUM_CPUS=${RAY_NUM_CPUS}, REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS}, AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS}"
    echo "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}, MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
    "${CMD[@]}" "$@"
} 2>&1 | tee -a "${LOG_FILE}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

cleanup_live_plot
PLOT_PID=""

if [[ "${TRAIN_STATUS}" == "0" ]]; then
    "${PYTHON_BIN}" scripts/plot_training_metrics.py "${LOG_FILE}" || true
fi

exit "${TRAIN_STATUS}"
