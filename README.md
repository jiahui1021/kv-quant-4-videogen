# KV-cache quantization baselines for video diffusion

RTN, KIVI and QuaRot KV-cache quantizers for Self-Forcing, Causal-Forcing and
LongCat-Video, plus the official QVG baseline on Causal-Forcing. One
implementation in [`kv_quant/`](kv_quant/), three model integrations.

## Methods

| Method | Quantizer |
|---|---|
| `BF16` | uncompressed reference |
| `RTN_INT4` / `RTN_INT2` | per-token asymmetric round-to-nearest over channel groups of 64 |
| `KIVI_INT4` / `KIVI_INT2` | keys per channel over token groups of 32, values per token, 128-token BF16 residual |
| `QUAROT_KV_INT4` / `QUAROT_KV_INT2` | post-RoPE Hadamard on Q/K, head-sized Hadamard on V undone on the attention output, then asymmetric groups of 64 with clip ratio 0.95 |
| `HADAMARD_K_INT4` / `HADAMARD_K_INT2` | QuaRot without the V rotation; an ablation, not a baseline |
| `QVG_INT2` / `QVG_INT4` | official Quant-VideoGen, Causal-Forcing only |

Every row stores its quantization parameters the same way: groups of 64
values with a BF16 scale and a BF16 zero, half a bit of metadata per value.
The three baselines therefore sit at the same compression and differ only by
the quantizer.

| | effective bits | compression |
|---|---:|---:|
| INT4 | 4.50 | 3.56x |
| INT2 | 2.50 | 6.40x |

KIVI's 128-token BF16 residual adds 0.005 bits over a 180-frame cache.

`--block_size` overrides the group size for every method, and
`--kv_channel_group_size 128` puts QuaRot on its own paper's KV setting. FP8 is
not used anywhere: the A100s these runs target have no FP8.

Measured on a Self-Forcing trace at matched compression, attention-output
error: INT4 0.007 (KIVI), 0.052 (QuaRot), 0.064 (RTN); INT2 0.107 (KIVI),
0.416 (QuaRot), 0.532 (RTN).

## Generation protocol

Prompts, seeds, attention window and horizon match the Tempokv and QVG
runners, so videos are comparable across the three repositories: MovieGen-128
prompts, noise seed `seed + prompt_index * 1000003`, full-history attention,
180 latent frames (717 pixel frames at 16 fps).

## Layout

```text
kv_quant/                     the quantizers, shared by all three models
Self-Forcing/scripts/         generate, evaluate, summarize
Causal-Forcing/               integration, inference.py, launchers
LongCat/                      integration, run_long_t2v.py, launcher
scripts/paper_experiments/    efficiency and latency runners
third_party/Quant-VideoGen/   vendored QVG
```

## Install

```bash
pip install -r Self-Forcing/requirements-inference.txt   # generation
pip install -r Self-Forcing/requirements-eval.txt        # VBench and fidelity
pip install -r Causal-Forcing/requirements.txt           # Causal-Forcing
pip install -r Causal-Forcing/requirements-qvg.txt       # QVG only
```

## Self-Forcing

```bash
python Self-Forcing/scripts/01_generate.py \
  --method RTN_INT4 \
  --checkpoint-path /path/to/self_forcing_dmd.pt \
  --results-root results
```

All six baseline rows at INT4 and INT2, against an existing BF16 run:

```bash
BF16_DIR=results/videos/BF16 bash Self-Forcing/scripts/07_run_paper_baselines.sh
```

## Causal-Forcing

```bash
python Causal-Forcing/inference.py \
  --config_path Causal-Forcing/configs/causal_forcing_dmd_framewise.yaml \
  --checkpoint_path /path/to/causal_forcing.pt \
  --data_path Self-Forcing/prompts/moviegen_128.txt \
  --output_folder results/causal_forcing/RTN_INT4 \
  --num_output_frames 180 \
  --method RTN_INT4
```

The whole matrix: `CONFIG_PATH=... CHECKPOINT_PATH=... DATA_PATH=...
bash Causal-Forcing/run_baseline_matrix.sh`. QVG has its own launcher,
`run_qvg.sh`, and needs `requirements-qvg.txt`.

## LongCat

Every continuation run starts from one shared initial video:

```bash
torchrun --nproc_per_node=1 LongCat/run_long_t2v.py \
  --workload 480p_init --method BF16 --quant_type none \
  --checkpoint_dir /path/to/LongCat-checkpoint \
  --output_dir results/longcat_init \
  --prompt "A person walking through a sunlit forest"

torchrun --nproc_per_node=1 LongCat/run_long_t2v.py \
  --workload 480p_long_gen --method RTN_INT4 --quant_type none \
  --no_offload_kv_cache \
  --checkpoint_dir /path/to/LongCat-checkpoint \
  --init_video_path results/longcat_init/0-0.mp4 \
  --num_segments 8 --seed 0 \
  --output_dir results/longcat/RTN_INT4 \
  --prompt "A person walking through a sunlit forest"
```

Change only `--method` between rows. The matrix launcher is
`LongCat/run_baseline_matrix.sh`.

## Metrics

Each run writes `metrics/efficiency_<METHOD>.json` (Self-Forcing) or one
report per run (Causal-Forcing, LongCat) with runtime, peak VRAM and the
cache bytes. Compression is `resident_analytic.v1`: the BF16 cost of the cache
positions actually held against the bytes resident at the same moment, never
the preallocated capacity.

Evaluation: `02_eval_fidelity.py` (PSNR/SSIM/LPIPS against BF16),
`03_eval_vbench.sh` (VBench), `05_summarize_results.py` (tables).
