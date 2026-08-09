#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the LongCat checkpoint directory}"
: "${INIT_VIDEO_PATH:?Set INIT_VIDEO_PATH to the shared initial video}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/longcat}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_SEGMENTS="${NUM_SEGMENTS:-8}"

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
  echo "===== LongCat ${method} ====="
  torchrun --nproc_per_node="${NPROC_PER_NODE}" LongCat/run_long_t2v.py \
    --workload 480p_long_gen \
    --context_parallel_size 1 \
    --method "${method}" \
    --block_size 16 \
    --quant_type none \
    --no_offload_kv_cache \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --init_video_path "${INIT_VIDEO_PATH}" \
    --num_segments "${NUM_SEGMENTS}" \
    --output_dir "${OUTPUT_ROOT}/${method}"
done
