#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE="${WHEELHOUSE:-/media/iie/4Tb/zccl/verl_vllm_wheels}"
PYTHON_BIN="${PYTHON_BIN:-/home/iie/miniconda3/envs/rl_study/bin/python}"

REQUIRED_WHEELS=(
    "vllm-0.8.5-*.whl"
    "torch-2.6.0-*.whl"
    "torchvision-0.21.0-*.whl"
    "torchaudio-2.6.0-*.whl"
    "xformers-0.0.29.post2-*.whl"
    "tensordict-*.whl"
)

if [[ ! -d "${WHEELHOUSE}" ]]; then
    echo "ERROR: missing wheelhouse directory: ${WHEELHOUSE}" >&2
    exit 2
fi

missing=0
for pattern in "${REQUIRED_WHEELS[@]}"; do
    if ! compgen -G "${WHEELHOUSE}/${pattern}" >/dev/null; then
        echo "MISSING: ${pattern}" >&2
        missing=1
    fi
done

if [[ "${missing}" != "0" ]]; then
    echo "Download the wheelhouse first, then rerun this script." >&2
    exit 2
fi

"${PYTHON_BIN}" -m pip install \
    --no-index \
    --find-links="${WHEELHOUSE}" \
    "vllm==0.8.5" \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    "xformers==0.0.29.post2" \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0"

"${PYTHON_BIN}" - <<'PY'
import tensordict
import torch
import vllm

print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"tensordict={tensordict.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
