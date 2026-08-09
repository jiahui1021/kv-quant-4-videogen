#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG_PATH:?Set CONFIG_PATH to the Causal-Forcing config}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the Causal-Forcing checkpoint}"
: "${DATA_PATH:?Set DATA_PATH to the prompt file}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/causal_forcing}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
PYTHON_BIN="${PYTHON_BIN:-python}"

METHODS=(
  BF16
  RTN_INT4
  RTN_INT2
  KIVI_INT4
  KIVI_INT2
  QUAROT_KV_INT4
  QUAROT_KV_INT2
)

for method in "${METHODS[@]}"; do
  echo "===== Causal-Forcing ${method} ====="
  "${PYTHON_BIN}" Causal-Forcing/inference.py \
    --config_path "${CONFIG_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_folder "${OUTPUT_ROOT}/${method}" \
    --num_output_frames "${NUM_OUTPUT_FRAMES}" \
    --method "${method}" \
    --block_size 16 \
    --use_ema
done
