#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/zjt/qwen3vl}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_qwenvl.sh"

rs-vlm-api &
API_PID=$!
trap 'kill ${API_PID}' EXIT
sleep 3
rs-vlm-demo
