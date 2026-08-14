# KV Quantization for LongCat and Causal-Forcing

This repository provides one shared implementation of KV-cache quantization for LongCat and Causal-Forcing. The shared baselines are RTN, KIVI, and KV-only QuaRot; Causal-Forcing additionally exposes the official QVG INT2 and INT4 baselines.

[中文说明](README.zh-CN.md)

## Supported methods

| Method | Bits | Description |
|---|---:|---|
| `BF16` | 16 | Full-precision KV cache |
| `RTN_INT4` / `RTN_INT2` | 4 / 2 | Round-to-nearest symmetric quantization |
| `KIVI_INT4` / `KIVI_INT2` | 4 / 2 | KIVI-style asymmetric key/value quantization |
| `QUAROT_KV_INT4` / `QUAROT_KV_INT2` | 4 / 2 | Hadamard rotation followed by KV quantization |
| `QVG_INT2` / `QVG_INT4` | 2 / 4 | Official Quant-VideoGen semantic smoothing + progressive residual quantization (Causal-Forcing only) |

The shared quantizers use `block_size=16` by default. QVG keeps the official
`quant_block_size=64` and its own eight-chunk schedule. LongCat and
Causal-Forcing call the shared implementation in [`kv_quant/`](kv_quant/);
QVG-specific code stays in the Causal-Forcing adapter.

## Directory layout

```text
kv-quant-4-videogen/
├── kv_quant/                         # Shared RTN/KIVI/QuaRot implementation
├── LongCat/
│   ├── kv_quant_adapter.py           # LongCat [B,H,S,D] layout adapter
│   ├── run_long_t2v.py               # LongCat generation entry point
│   └── run_baseline_matrix.sh        # Run the shared seven methods
├── Causal-Forcing/
│   ├── kv_quant_runtime.py           # Causal cache setup and reset helpers
│   ├── inference.py                  # Causal generation entry point
│   ├── qvg_runtime.py                # Official QVG adapter and schedule
│   └── run_baseline_matrix.sh        # Run shared methods plus QVG_INT2/INT4
├── third_party/Quant-VideoGen/       # Pinned official QVG codec
└── Self-Forcing/                     # Existing Self-Forcing implementation
```

## Requirements

The generation commands require a Linux environment with an NVIDIA GPU, a CUDA-enabled PyTorch installation, and the original model dependencies.

For Causal-Forcing:

```bash
cd Causal-Forcing
conda create -n kv-quant python=3.10 -y
conda activate kv-quant
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop
cd ..
```

LongCat additionally needs the dependencies used by its original runtime, including `diffusers`, `transformers`, `accelerate`, `safetensors`, `einops`, `triton`, `torchvision`, `Pillow`, `loguru`, `ftfy`, `regex`, `openai`, and `termcolor`. Install versions compatible with the installed CUDA and PyTorch versions.

The LongCat source imports the existing `quant_videogen` runtime. If that runtime is kept outside this repository, expose it before running LongCat:

```bash
export PYTHONPATH=/path/to/Quant-VideoGen:$PYTHONPATH
```

Replace the path with the directory that contains the `quant_videogen/` package.

## LongCat usage

### 1. Generate one shared initial video

LongCat continuation experiments must use the same initial video for every method. Generate it once with `BF16`, then reuse the resulting file:

```bash
torchrun --nproc_per_node=1 LongCat/run_long_t2v.py \
  --workload 480p_init \
  --context_parallel_size 1 \
  --method BF16 \
  --block_size 16 \
  --quant_type none \
  --checkpoint_dir /path/to/LongCat-checkpoint \
  --output_dir results/longcat_init \
  --prompt "A person walking through a sunlit forest"
```

With the default `--prompt_idx 0` and `--seed 0`, the initial video is written as `results/longcat_init/0-0.mp4`.

### 2. Run one method

```bash
torchrun --nproc_per_node=1 LongCat/run_long_t2v.py \
  --workload 480p_long_gen \
  --context_parallel_size 1 \
  --method RTN_INT4 \
  --block_size 16 \
  --quant_type none \
  --no_offload_kv_cache \
  --checkpoint_dir /path/to/LongCat-checkpoint \
  --init_video_path results/longcat_init/0-0.mp4 \
  --num_segments 8 \
  --num_frames 93 \
  --num_cond_frames 53 \
  --seed 0 \
  --output_dir results/longcat/RTN_INT4 \
  --prompt "A person walking through a sunlit forest"
```

Change only `--method` to compare the seven shared baselines. Keep the prompt, seed, initial video, frame settings, and context-parallel settings unchanged.

For lower GPU memory usage, replace `--no_offload_kv_cache` with `--offload_kv_cache`.

Do not combine the shared method with LongCat's legacy `--quant_type` quantizer. Use `--quant_type none` for all seven methods. This combination is invalid:

```bash
--method RTN_INT2 --quant_type naive-int2
```

### 3. Run the complete LongCat matrix

```bash
CHECKPOINT_DIR=/path/to/LongCat-checkpoint \
INIT_VIDEO_PATH=results/longcat_init/0-0.mp4 \
OUTPUT_ROOT=results/longcat \
NPROC_PER_NODE=1 \
bash LongCat/run_baseline_matrix.sh
```

The script runs:

```text
BF16 RTN_INT4 RTN_INT2 KIVI_INT4 KIVI_INT2 QUAROT_KV_INT4 QUAROT_KV_INT2
```

## Causal-Forcing usage

### 1. Run one method

```bash
python Causal-Forcing/inference.py \
  --config_path Causal-Forcing/configs/causal_forcing_dmd_framewise.yaml \
  --checkpoint_path /path/to/causal_forcing.pt \
  --data_path Causal-Forcing/prompts/demos.txt \
  --output_folder results/causal_forcing/RTN_INT4 \
  --num_output_frames 180 \
  --method RTN_INT4 \
  --block_size 16 \
  --use_ema
```

`--num_output_frames` is in latent frames; 180 latent frames produce 717
pixel frames (44.8s @ 16fps), matching the long-video causal_forcing results.

For QVG, install the dependencies in `Causal-Forcing/requirements-qvg.txt`
and run the launcher:

```bash
CHECKPOINT_PATH=/path/to/causal_forcing.pt \
DATA_PATH=Causal-Forcing/prompts/demos.txt \
bash Causal-Forcing/run_qvg.sh
```

The default is QVG_INT2; use `METHOD=QVG_INT4` for INT4. The launcher uses the
chunkwise 180-latent-frame configuration (717 pixel frames) and currently
supports T2V only.

For text-to-video, use a prompt file such as `Causal-Forcing/prompts/demos.txt`. For image-to-video, add `--i2v` and pass an image-prompt dataset supported by the original Causal-Forcing loader.

### 2. Run the complete Causal-Forcing matrix

```bash
CONFIG_PATH=Causal-Forcing/configs/causal_forcing_dmd_framewise.yaml \
CHECKPOINT_PATH=/path/to/causal_forcing.pt \
DATA_PATH=Causal-Forcing/prompts/demos.txt \
OUTPUT_ROOT=results/causal_forcing \
NUM_OUTPUT_FRAMES=180 \
LOCAL_ATTN_SIZE=51 \
bash Causal-Forcing/run_baseline_matrix.sh
```

The matrix script uses `--use_ema`. Remove that option from the script when using a checkpoint without EMA weights.

## Generation metrics

Each completed video writes a JSON report containing at least:

```json
{
  "method": "RTN_INT4",
  "end_to_end_generation_time_s": 0.0,
  "peak_vram_bytes": 0,
  "peak_vram_gb": 0.0,
  "quantize_calls": 0,
  "dequantize_calls": 0
}
```

LongCat reports are stored under:

```text
<output_dir>/<prompt_idx>-<seed>/metrics_<method>.json
```

Causal-Forcing reports are stored under:

```text
<output_folder>/metrics_<method>_<prompt_idx>.json
```

`end_to_end_generation_time_s` covers the generation call, decoding, and video writing. `peak_vram_bytes` is the maximum CUDA memory allocated during that video generation. Quantized runs also fail if the quantizer was never called, preventing a mislabeled BF16 run.

Per-quantizer timing is disabled by default so it does not add synchronization or event overhead to the latency benchmark. Enable the optional CUDA-event breakdown only when needed:

```bash
--profile_quant_timing        # LongCat and Causal-Forcing
--profile-quant-timing        # Self-Forcing
```

Causal-Forcing generic baselines report resident cache capacity. QVG reports
physical BF16 chunks, centroids, cluster IDs, packed residuals, scales and
zero-points separately, together with `resident_total_kv_bytes`,
`uncompressed_reference_kv_bytes`, effective bits/value, QVG configuration and
the pinned upstream commit. QVG resident bytes include both packed tensors and
the BF16 tail.

## Recommended validation order

```text
BF16 → RTN_INT4 → KIVI_INT4 → QUAROT_KV_INT4 → RTN_INT2 → KIVI_INT2 → QUAROT_KV_INT2 → QVG_INT2 → QVG_INT4
```

Start with `context_parallel_size=1`, no compilation, and the same initial video. Enable CPU offload, context parallelism, attention sinks, and local attention only after the basic matrix is working.

## Troubleshooting

- `ModuleNotFoundError: quant_videogen`: add the directory containing `quant_videogen/` to `PYTHONPATH`.
- `Do not enable shared RTN/KIVI/QuaRot and legacy --quant_type`: set `--quant_type none`.
- `KV quantization was never triggered`: use a cache-enabled LongCat continuation workload or the Causal-Forcing inference entry point, and confirm that the selected method is not `BF16`.
- Out-of-memory during LongCat generation: use `--offload_kv_cache`, reduce `--num_segments`, or validate one segment first.
