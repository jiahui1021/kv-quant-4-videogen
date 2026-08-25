#!/usr/bin/env bash
# Final Self-Forcing launcher for RTN, KIVI, and QuaRot-KV INT2/INT4.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
selfforcing_root="$(cd -- "$script_dir/../../.." && pwd -P)"
repo_root="$(cd -- "$selfforcing_root/.." && pwd -P)"
tempokv_root="${TEMPOKV_ROOT:-$(cd -- "$repo_root/.." && pwd -P)/Tempokv}"
python_bin="${PYTHON_BIN:-$repo_root/../Tempokv/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="${PYTHON_BIN:-python3}"
generate="$selfforcing_root/scripts/01_generate.py"
probe="$selfforcing_root/scripts/probe_horizon.py"
mode="${EFFICIENCY_MODE:-both}"
only="${ONLY:-}"
skip="${SKIP:-}"
output_root="${OUTPUT_ROOT:-$selfforcing_root/results/paper_experiments/efficiency-self-forcing/kvquant}"
checkpoint="${CHECKPOINT_PATH:-$selfforcing_root/checkpoints/self_forcing_dmd.pt}"
prompts="${PROMPT_PATH:-$selfforcing_root/prompts/MovieGenVideoBench_extended.txt}"
config="${CONFIG_PATH:-}"
default_config="${DEFAULT_CONFIG_PATH:-}"
frames="${FRAMES:-717}"
horizon_frames="${HORIZON_FRAMES:-3009}"
seed="${SEED:-0}"
dry_run=0
overwrite=0
collect=1

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'EOF'
Usage: Self-Forcing/scripts/paper_experiments/efficiency/run_efficiency.sh [options]

Required for a real run: CONFIG_PATH and DEFAULT_CONFIG_PATH, plus a valid
checkpoint and prompt file. The script runs six rows: RTN/KIVI/QuaRot-KV at
INT2 and INT4. `efficiency.csv` reports resident compression locally.
EOF
}
while (($#)); do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    --only) only="$2"; shift 2 ;;
    --skip) skip="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --prompts) prompts="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --default-config) default_config="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --horizon-frames) horizon_frames="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --overwrite) overwrite=1; shift ;;
    --no-collect) collect=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
case "$mode" in efficiency|horizon|both) ;; *) die "--mode must be efficiency, horizon, or both" ;; esac
(( (frames + 3) % 12 == 0 )) || die "FRAMES must satisfy (FRAMES + 3) % 12 == 0"
(( (horizon_frames + 3) % 12 == 0 )) || die "HORIZON_FRAMES must satisfy (FRAMES + 3) % 12 == 0"
mkdir -p -- "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"
latent_frames=$(( (frames + 3) / 4 ))
horizon_latent_frames=$(( (horizon_frames + 3) / 4 ))
contains() { [[ ",${1:-}," == *",$2,"* ]]; }
selected() {
  if [[ -n "$only" ]] && ! contains "$only" "$1"; then return 1; fi
  if [[ -n "$skip" ]] && contains "$skip" "$1"; then return 1; fi
}
require_real_inputs() {
  (( dry_run )) && return 0
  [[ -f "$checkpoint" ]] || die "checkpoint does not exist: $checkpoint"
  [[ -f "$prompts" ]] || die "prompt file does not exist: $prompts"
  [[ -n "$config" && -f "$config" ]] || die "set CONFIG_PATH to the quantizer config"
  [[ -n "$default_config" && -f "$default_config" ]] || die "set DEFAULT_CONFIG_PATH to the Self-Forcing default config"
}
check_record() {
  local name="$1" record="$2"
  if [[ -f "$output_root/$name/$record" && "$overwrite" != 1 ]]; then
    die "$name already has $record; use --overwrite or --skip $name"
  fi
}
run_cmd() {
  local label="$1"; shift
  printf '[%s] $' "$label"; printf ' %q' "$@"; printf '\n'
  (( dry_run )) && return 0
  "$@"
}
run_row() {
  local name="$1" bits="$2" base="${name%_INT*}" out="$output_root/$1"
  selected "$name" || return 0
  check_record "$name" efficiency.json
  local -a command=(
    "$python_bin" "$generate"
    --method "$name" --bits "$bits"
    --num-output-frames "$latent_frames"
    --local-attn-size "$latent_frames"
    --max-prompts 4 --seed "$seed"
    --profile-quant-timing --no-log-vram-trace
    --config-path "$config" --default-config-path "$default_config"
    --checkpoint-path "$checkpoint" --prompt-path "$prompts"
    --results-root "$out" --efficiency-output "$out/efficiency.json"
  )
  run_cmd "$name" "${command[@]}"
}
run_horizon_row() {
  local name="$1" bits="$2" out="$output_root/$1"
  selected "$name" || return 0
  check_record "$name" horizon.json
  local -a command=(
    env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    "$python_bin" "$probe" --method "$name" --probe-frames "$horizon_frames"
    --config-path "$config" --default-config-path "$default_config"
    --checkpoint-path "$checkpoint" --prompt-path "$prompts" --seed "$seed"
    --output "$out/horizon.json"
  )
  run_cmd "$name horizon" "${command[@]}"
}
require_real_inputs
if [[ "$mode" == efficiency || "$mode" == both ]]; then
  run_row RTN_INT2 2
  run_row RTN_INT4 4
  run_row KIVI_INT2 2
  run_row KIVI_INT4 4
  run_row QUAROT_KV_INT2 2
  run_row QUAROT_KV_INT4 4
fi
if [[ "$mode" == horizon || "$mode" == both ]]; then
  run_horizon_row RTN_INT2 2
  run_horizon_row RTN_INT4 4
  run_horizon_row KIVI_INT2 2
  run_horizon_row KIVI_INT4 4
  run_horizon_row QUAROT_KV_INT2 2
  run_horizon_row QUAROT_KV_INT4 4
fi
if (( ! dry_run && collect )); then
  collector="$tempokv_root/scripts/paper_experiments/efficiency/collect_efficiency.py"
  [[ -f "$collector" ]] || die "collector not found: $collector; set TEMPOKV_ROOT"
  "$python_bin" "$collector" --output-root "$output_root" --local-only
fi
