# LongCat 与 Causal-Forcing 的 KV 量化

本仓库为 LongCat 和 Causal-Forcing 提供统一的 KV-cache 量化实现，支持 RTN、KIVI 和 KV-only QuaRot；Causal-Forcing 额外提供官方 QVG INT2 和 INT4 baseline。

[English README](README.md)

## 支持的方法

| 方法 | 位宽 | 说明 |
|---|---:|---|
| `BF16` | 16 | 完整精度 KV cache |
| `RTN_INT4` / `RTN_INT2` | 4 / 2 | Round-to-nearest 对称量化 |
| `KIVI_INT4` / `KIVI_INT2` | 4 / 2 | KIVI 风格的 K/V 非对称量化 |
| `QUAROT_KV_INT4` / `QUAROT_KV_INT2` | 4 / 2 | Hadamard 旋转后进行 KV 量化 |
| `QVG_INT2` / `QVG_INT4` | 2 / 4 | 官方 Quant-VideoGen semantic smoothing + progressive residual quantization（仅 Causal-Forcing） |

共享量化器默认使用 `block_size=16`；QVG 保留官方 `quant_block_size=64` 和每 8 个
generation chunk 的独立 schedule。LongCat 和 Causal-Forcing 共同调用根目录
[`kv_quant/`](kv_quant/) 中的共享实现，QVG 专用代码放在 Causal-Forcing adapter 中。

## 目录结构

```text
kv-quant-4-videogen/
├── kv_quant/                         # 共享 RTN/KIVI/QuaRot 实现
├── LongCat/
│   ├── kv_quant_adapter.py           # LongCat [B,H,S,D] layout 适配器
│   ├── run_long_t2v.py               # LongCat 生成入口
│   └── run_baseline_matrix.sh        # 批量运行共享的 7 种方法
├── Causal-Forcing/
│   ├── kv_quant_runtime.py           # Causal cache 初始化与 reset
│   ├── inference.py                  # Causal 生成入口
│   ├── qvg_runtime.py                # 官方 QVG adapter 与 schedule
│   └── run_baseline_matrix.sh        # 运行共享方法和 QVG_INT2/INT4
├── third_party/Quant-VideoGen/       # 固定版本的官方 QVG codec
└── Self-Forcing/                     # 原有 Self-Forcing 实现
```

## 环境要求

视频生成需要 Linux、NVIDIA GPU、支持 CUDA 的 PyTorch，以及两个原始模型项目所需的依赖。

### Causal-Forcing 环境

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

LongCat 还需要原始运行环境中的 `diffusers`、`transformers`、`accelerate`、`safetensors`、`einops`、`triton`、`torchvision`、`Pillow`、`loguru`、`ftfy`、`regex`、`openai` 和 `termcolor`。请安装与当前 CUDA、PyTorch 相匹配的版本。

LongCat 源码会导入已有的 `quant_videogen` 运行时。如果该运行时放在当前仓库之外，需要在运行前设置：

```bash
export PYTHONPATH=/path/to/Quant-VideoGen:$PYTHONPATH
```

这里的路径必须指向包含 `quant_videogen/` 的目录。

## LongCat 使用方法

### 1. 生成统一的初始视频

LongCat continuation 实验必须让所有方法使用同一个初始视频。先用 `BF16` 生成一次，后续方法都复用这个文件：

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

默认 `--prompt_idx 0`、`--seed 0` 时，初始视频位置为 `results/longcat_init/0-0.mp4`。

### 2. 运行单个方法

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

只替换 `--method` 即可比较 7 种共享 baseline。prompt、seed、初始视频、帧数设置和 context parallel 设置必须保持一致。

显存不足时可以把 `--no_offload_kv_cache` 换成 `--offload_kv_cache`。

共享方法不能和 LongCat 原有的 `--quant_type` 同时使用。7 种方法都应设置 `--quant_type none`。下面的组合无效：

```bash
--method RTN_INT2 --quant_type naive-int2
```

### 3. 批量运行 LongCat baseline

```bash
CHECKPOINT_DIR=/path/to/LongCat-checkpoint \
INIT_VIDEO_PATH=results/longcat_init/0-0.mp4 \
OUTPUT_ROOT=results/longcat \
NPROC_PER_NODE=1 \
bash LongCat/run_baseline_matrix.sh
```

脚本会依次运行：

```text
BF16 RTN_INT4 RTN_INT2 KIVI_INT4 KIVI_INT2 QUAROT_KV_INT4 QUAROT_KV_INT2
```

## Causal-Forcing 使用方法

### 1. 运行单个方法

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

`--num_output_frames` 是 latent 帧数；180 latent 帧对应 717 像素帧
（44.8s @ 16fps），与长视频 causal_forcing 结果对齐。

文生视频使用 `Causal-Forcing/prompts/demos.txt` 等 prompt 文件。图生视频使用 `--i2v`，并传入原 Causal-Forcing loader 支持的图像 prompt 数据集。

frame-wise 和 chunk-wise 模型通过 `--config_path` 选择，量化参数保持不变。

### 2. QVG_INT2 / QVG_INT4

安装 [`Causal-Forcing/requirements-qvg.txt`](Causal-Forcing/requirements-qvg.txt)
中的依赖后，直接运行 launcher：

```bash
CHECKPOINT_PATH=/path/to/causal_forcing.pt \
DATA_PATH=Causal-Forcing/prompts/demos.txt \
bash Causal-Forcing/run_qvg.sh
```

默认运行 QVG_INT2；设置 `METHOD=QVG_INT4` 可运行 INT4。launcher 默认使用正式
chunkwise workload：717 pixel frames、180 latent frames、180-frame full-history attention，
仅支持 T2V。

### 3. 批量运行 Causal-Forcing baseline

```bash
CONFIG_PATH=Causal-Forcing/configs/causal_forcing_dmd_chunkwise.yaml \
CHECKPOINT_PATH=/path/to/causal_forcing.pt \
DATA_PATH=Causal-Forcing/prompts/demos.txt \
OUTPUT_ROOT=results/causal_forcing \
bash Causal-Forcing/run_baseline_matrix.sh
```

checkpoint 包含 EMA 权重时设置 `USE_EMA=1`；正式 launcher 默认使用普通
`generator` 权重，与 Tempokv launcher 保持一致。
只有需要量化器 CUDA event 分解时才设置 `PROFILE_QUANT_TIMING=1`；为保证 latency
公平比较，默认关闭。

## 生成指标

每个完成的视频都会写入 JSON，至少包含：

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

LongCat 指标位置：

```text
<output_dir>/<prompt_idx>-<seed>/metrics_<method>.json
```

Causal-Forcing 指标位置：

```text
<output_folder>/metrics_<method>_<prompt_idx>.json
```

`diffusion_generation_s` 是公平比较的主延迟指标：从自回归 denoising loop 开始，
到最后一个 clean refresh 完成结束，不包含 VAE decode。`end_to_end_generation_time_s`
保留用于兼容旧结果，并包含 VAE decode。`peak_vram_bytes` 是该视频生成过程中的最高
CUDA 已分配显存。量化方法如果从未实际触发量化，会直接报错，避免把 BF16 结果误标成量化结果。

为避免影响 latency benchmark，量化器的逐次耗时统计默认关闭，不会在每次量化或反量化时插入同步或事件开销。需要查看 CUDA event 耗时分解时再显式开启：

```bash
--profile_quant_timing        # LongCat 和 Causal-Forcing
--profile-quant-timing        # Self-Forcing
```

Causal-Forcing 通用 baseline 按 resident cache capacity 统计；QVG 额外分别报告
BF16 chunk、centroid、cluster ID、packed residual、scale 和 zero-point 的物理显存，
并输出 `resident_total_kv_bytes`、`uncompressed_reference_kv_bytes`、effective bits/value、
QVG 配置和固定 upstream commit；resident bytes 包含压缩数据和 BF16 尾部。
effective bits/value 的计算为
`resident_total_kv_bytes * 8 / resident_logical_kv_values`，表示实际驻留存储，
不等同于 nominal INT2/INT4 位宽。

## 推荐验证顺序

```text
BF16 → RTN_INT4 → KIVI_INT4 → QUAROT_KV_INT4 → RTN_INT2 → KIVI_INT2 → QUAROT_KV_INT2 → QVG_INT2 → QVG_INT4
```

第一阶段固定 `context_parallel_size=1`、关闭 compilation，并让所有方法使用同一个初始视频。基础矩阵通过后，再逐项开启 CPU offload、context parallel、attention sink 和 local attention。

## 常见问题

- `ModuleNotFoundError: quant_videogen`：将包含 `quant_videogen/` 的目录加入 `PYTHONPATH`。
- `Do not enable shared RTN/KIVI/QuaRot and legacy --quant_type`：设置 `--quant_type none`。
- `KV quantization was never triggered`：LongCat 使用支持 cache 的 continuation workload；Causal-Forcing 使用 `inference.py`；并确认方法不是 `BF16`。
- LongCat 显存不足：使用 `--offload_kv_cache`，减少 `--num_segments`，或先只验证一个 segment。
