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
  QVG_INT2
  QVG_INT4
)

for method in "${METHODS[@]}"; do
  echo "===== Causal-Forcing ${method} ====="
  if [[ "${method}" == QVG_* ]]; then
    method_args=(
      --method "${method}"
      --qvg_quant_factor 8
      --qvg_num_k_centroids 256
      --qvg_num_v_centroids 256
      --qvg_kmeans_max_iters 2
      --qvg_quant_block_size 64
      --qvg_num_prq_stages 1
    )
  else
    method_args=(--method "${method}" --block_size 16)
  fi
  "${PYTHON_BIN}" Causal-Forcing/inference.py \
    --config_path "${CONFIG_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_folder "${OUTPUT_ROOT}/${method}" \
    --num_output_frames "${NUM_OUTPUT_FRAMES}" \
    "${method_args[@]}" \
    --use_ema
done
