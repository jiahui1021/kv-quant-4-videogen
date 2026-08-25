#!/usr/bin/env bash
# Run one 717-pixel-frame Forcing-KV video on Self-Forcing and Causal-Forcing.
# Each run writes efficiency/compression_ratio_0.json.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
forcing_kv_root="$(cd -- "$script_dir/.." && pwd -P)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$forcing_kv_root/../../Tempokv/.venv/bin/python" ]]; then
  python_bin="$forcing_kv_root/../../Tempokv/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
else
  python_bin="$(command -v python3 || true)"
fi
pretrained_root="${FORCING_KV_PRETRAINED_ROOT:-}"
common_checkpoint="${CHECKPOINT_PATH:-}"
sf_checkpoint="${SF_CHECKPOINT_PATH:-$common_checkpoint}"
cf_checkpoint="${CF_CHECKPOINT_PATH:-$common_checkpoint}"
sf_config="${SF_CONFIG_PATH:-$forcing_kv_root/configs/forcing-kv/forcingkv_self_forcing_inference.yaml}"
cf_config="${CF_CONFIG_PATH:-$forcing_kv_root/configs/forcing-kv/forcingkv_causal_forcing_inference.yaml}"
sf_prompts="${SF_PROMPTS_PATH:-$forcing_kv_root/prompts/example_prompts.txt}"
cf_prompts="${CF_PROMPTS_PATH:-$forcing_kv_root/prompts/example_prompts.txt}"
output_root="${OUTPUT_ROOT:-$forcing_kv_root/results/sf_cf_717}"
seed="${SEED:-0}"
prompt_index="${PROMPT_INDEX:-0}"
only="${ONLY:-both}"
dry_run=0

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'EOF'
Usage: Forcing-KV/scripts/run_sf_cf_717.sh [options]

Required for a real run:
  FORCING_KV_PRETRAINED_ROOT  directory containing Wan2.1-T2V-1.3B/
  SF_CHECKPOINT_PATH / CF_CHECKPOINT_PATH, or CHECKPOINT_PATH for both

Options:
  --only sf|cf|both       Workloads to run (default: both)
  --output-root PATH      Output directory
  --dry-run               Print commands without loading models
  --prompt-index N        Prompt index, default 0
EOF
}

while (($#)); do
  case "$1" in
    --only) only="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --prompt-index) prompt_index="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
case "$only" in sf|cf|both) ;; *) die "--only must be sf, cf, or both" ;; esac

if (( ! dry_run )); then
  [[ -n "$pretrained_root" && -d "$pretrained_root/Wan2.1-T2V-1.3B" ]] || \
    die "FORCING_KV_PRETRAINED_ROOT must contain Wan2.1-T2V-1.3B"
  [[ -n "$sf_checkpoint" && -f "$sf_checkpoint" ]] || \
    die "SF_CHECKPOINT_PATH or CHECKPOINT_PATH must point to an existing file"
  [[ -n "$cf_checkpoint" && -f "$cf_checkpoint" ]] || \
    die "CF_CHECKPOINT_PATH or CHECKPOINT_PATH must point to an existing file"
  [[ -f "$sf_config" && -f "$cf_config" ]] || die "SF_CONFIG_PATH/CF_CONFIG_PATH is missing"
  [[ -f "$sf_prompts" && -f "$cf_prompts" ]] || die "prompt file is missing"
fi

mkdir -p -- "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"
model_name="${pretrained_root:-/path/to/pretrained}/Wan2.1-T2V-1.3B"

run_one() {
  local label="$1" config="$2" checkpoint="$3" prompts="$4"
  local output="$output_root/$label"
  local report="$output/efficiency/compression_ratio_${prompt_index}.json"
  local -a command=(
    env
    "PYTHONPATH=$forcing_kv_root"
    "FORCING_KV_PRETRAINED_ROOT=$pretrained_root"
    "$python_bin" "$forcing_kv_root/inference.py"
    --config_path "$config"
    --checkpoint_path "$checkpoint"
    --model_name "$model_name"
    --data_path "$prompts"
    --output_folder "$output"
    --num_output_frames 180
    --local_attn_size 180
    --num_samples 1
    --seed "$seed"
    --prompt_index "$prompt_index"
    --no-profile
  )
  printf '[%s] $' "$label"
  printf ' %q' "${command[@]}"
  printf '\n'
  if (( dry_run )); then
    return 0
  fi
  "${command[@]}"
  "$python_bin" "$forcing_kv_root/tools/forcing_kv_compression_ratio.py" \
    --report "$report"
}

if [[ "$only" == sf || "$only" == both ]]; then
  run_one self_forcing "$sf_config" "$sf_checkpoint" "$sf_prompts"
fi
if [[ "$only" == cf || "$only" == both ]]; then
  run_one causal_forcing "$cf_config" "$cf_checkpoint" "$cf_prompts"
fi
