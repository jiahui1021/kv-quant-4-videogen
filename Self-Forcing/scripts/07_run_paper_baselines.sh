#!/usr/bin/env bash
# Re-run the paper's three KV-cache baselines on Self-Forcing at INT4 and INT2.
#
# Unlike 06_run_baseline_matrix.sh this does NOT generate BF16: it reuses an
# existing BF16 run as the fidelity reference.  That reference must come from
# the same prompt file, seed and frame count as the runs below, or PSNR/SSIM
# compare videos of different content.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFER_PYTHON="${INFER_PYTHON:-python}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results}"
GPU_ID="${GPU_ID:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-60}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-180}"
SEED="${SEED:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/prompts/moviegen_128.txt}"

# Where the existing BF16 videos live.  Defaults to this run root, which is
# where 06_run_baseline_matrix.sh would have put them.
BF16_DIR="${BF16_DIR:-${RUN_ROOT}/videos/BF16}"

# QuaRot INT2 uses the paper's KV setting (asymmetric, group head_dim, clip 0.95),
# which is the quantizer default.  Extra QuaRot knobs can still be passed here.
QUAROT_INT2_ARGS="${QUAROT_INT2_ARGS:-}"

RUN_VBENCH="${RUN_VBENCH:-1}"
RUN_FIDELITY="${RUN_FIDELITY:-1}"

METHODS=(
  RTN_INT4
  RTN_INT2
  KIVI_INT4
  KIVI_INT2
  QUAROT_KV_INT4
  QUAROT_KV_INT2
)

if [[ "${RUN_FIDELITY}" == 1 ]]; then
  if [[ ! -d "${BF16_DIR}" ]]; then
    echo "BF16_DIR does not exist: ${BF16_DIR}" >&2
    echo "Point BF16_DIR at the existing BF16 videos, or set RUN_FIDELITY=0." >&2
    exit 2
  fi
  bf16_count="$(find "${BF16_DIR}" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
  if [[ "${bf16_count}" -eq 0 ]]; then
    echo "No .mp4 files under ${BF16_DIR}" >&2
    exit 2
  fi
  echo "[info] BF16 reference: ${BF16_DIR} (${bf16_count} videos)"
  # 02_eval_fidelity.py demands a candidate for every BF16 file unless told to
  # intersect.  A 60-prompt run against a larger BF16 set needs that flag.
  if [[ "${bf16_count}" -ne "${MAX_PROMPTS}" ]]; then
    echo "[info] BF16 set (${bf16_count}) differs from MAX_PROMPTS (${MAX_PROMPTS});"
    echo "[info] fidelity will run with --match-intersection."
    MATCH_ARGS=(--match-intersection)
  else
    MATCH_ARGS=()
  fi
fi

for method in "${METHODS[@]}"; do
  echo "===== generate ${method} ====="
  extra_args=()
  if [[ "${method}" == "QUAROT_KV_INT2" && -n "${QUAROT_INT2_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${QUAROT_INT2_ARGS})
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${INFER_PYTHON}" "${ROOT_DIR}/scripts/01_generate.py" \
    --method "${method}" \
    --block-size "${BLOCK_SIZE}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --use-ema \
    --prompt-path "${PROMPT_FILE}" \
    --max-prompts "${MAX_PROMPTS}" \
    --num-output-frames "${NUM_OUTPUT_FRAMES}" \
    --results-root "${RUN_ROOT}" \
    "${extra_args[@]}"
done

if [[ "${RUN_VBENCH}" == 1 ]]; then
  for method in "${METHODS[@]}"; do
    echo "===== vbench ${method} ====="
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHON_BIN="${EVAL_PYTHON}" RUN_ROOT="${RUN_ROOT}" \
      "${ROOT_DIR}/scripts/03_eval_vbench.sh" \
      "${method}" \
      "${RUN_ROOT}/videos/${method}" \
      "${PROMPT_FILE}" \
      "${RUN_ROOT}/metrics/vbench_${method}" \
      "${RUN_ROOT}/metrics/vbench_${method}.json"
  done
fi

if [[ "${RUN_FIDELITY}" == 1 ]]; then
  for method in "${METHODS[@]}"; do
    echo "===== fidelity ${method} ====="
    "${INFER_PYTHON}" "${ROOT_DIR}/scripts/02_eval_fidelity.py" \
      --bf16-dir "${BF16_DIR}" \
      --candidate-dir "${RUN_ROOT}/videos/${method}" \
      --output "${RUN_ROOT}/metrics/fidelity_${method}.json" \
      --device cpu \
      "${MATCH_ARGS[@]}"
  done
fi

"${INFER_PYTHON}" "${ROOT_DIR}/scripts/05_summarize_results.py" --results-root "${RUN_ROOT}"
