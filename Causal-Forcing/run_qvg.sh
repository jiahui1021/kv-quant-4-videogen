#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-Causal-Forcing/configs/causal_forcing_dmd_chunkwise_qvg.yaml}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the Causal-Forcing checkpoint}"
: "${DATA_PATH:?Set DATA_PATH to the prompt file}"

METHOD="${METHOD:-QVG_INT2}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-51}"
LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-51}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-results/qvg/${METHOD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

extra_args=()
if [[ "${QVG_DISABLE_COMPRESSION:-0}" == "1" ]]; then
  extra_args+=(--qvg_disable_compression)
fi

"${PYTHON_BIN}" Causal-Forcing/inference.py \
  --config_path "${CONFIG_PATH}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_folder "${OUTPUT_FOLDER}" \
  --num_output_frames "${NUM_OUTPUT_FRAMES}" \
  --local_attn_size "${LOCAL_ATTN_SIZE}" \
  --method "${METHOD}" \
  --qvg_quant_factor 8 \
  --qvg_num_k_centroids 256 \
  --qvg_num_v_centroids 256 \
  --qvg_kmeans_max_iters 2 \
  --qvg_quant_block_size 64 \
  --qvg_num_prq_stages 1 \
  --profile_quant_timing \
  --use_ema \
  "${extra_args[@]}"
