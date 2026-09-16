#!/usr/bin/env bash
# Activate the project virtualenv and expose its CUDA 12.1 runtime libraries.

RS_VLM_ROOT="${RS_VLM_ROOT:-/workspace/zjt/qwen3vl}"

# shellcheck disable=SC1091
source "${RS_VLM_ROOT}/qwenvl/bin/activate"

NVRTC_LIB="${VIRTUAL_ENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
CUDA_RUNTIME_LIB="${VIRTUAL_ENV}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"

export LD_LIBRARY_PATH="${NVRTC_LIB}:${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

unset NVRTC_LIB CUDA_RUNTIME_LIB
