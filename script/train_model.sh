#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <recipe.yaml> [num_processes]"
  exit 1
fi

RECIPE="$1"
NUM_PROCESSES="${2:-1}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  --num_processes "${NUM_PROCESSES}" \
  src/grpo.py \
  --config "${RECIPE}"
