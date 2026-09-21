# 视频扩散模型的 KV cache 量化 baseline

在 Self-Forcing、Causal-Forcing、LongCat-Video 上实现 RTN、KIVI、QuaRot 三种 KV cache
量化，Causal-Forcing 上另有官方 QVG。量化器只有一份，放在 [`kv_quant/`](kv_quant/)，
三个模型共用。

## 方法

| 方法 | 量化器 |
|---|---|
| `BF16` | 不压缩的对照 |
| `RTN_INT4` / `RTN_INT2` | per-token 非对称 round-to-nearest，64 通道一组 |
| `KIVI_INT4` / `KIVI_INT2` | K 按通道在 32 token 一组上量化，V 按 token 量化，保留 128 token 的 BF16 residual |
| `QUAROT_KV_INT4` / `QUAROT_KV_INT2` | RoPE 之后对 Q/K 做 Hadamard，V 做 head_dim 大小的旋转并在 attention 输出处还原，再按 64 通道一组非对称量化，clip ratio 0.95 |
| `HADAMARD_K_INT4` / `HADAMARD_K_INT2` | 去掉 V 旋转的 QuaRot，消融用，不作为 baseline |
| `QVG_INT2` / `QVG_INT4` | 官方 Quant-VideoGen，仅 Causal-Forcing |

三条 baseline 的量化参数都按同一种方式存：64 个值一组，一个 BF16 scale 加一个 BF16
zero，每个值半个 bit 的元数据。所以压缩比完全相同，行与行只差量化方法本身。

| | 有效位宽 | 压缩比 |
|---|---:|---:|
| INT4 | 4.50 | 3.56× |
| INT2 | 2.50 | 6.40× |

KIVI 的 128 token BF16 residual 在 180 帧的 cache 上只多 0.005 bit。

`--block_size` 可以统一改所有方法的分组，`--kv_channel_group_size 128` 把 QuaRot 切回
它自己论文的 KV 设置。全程不用 FP8，因为跑实验的 A100 不支持。

在 Self-Forcing 的 trace 上实测，压缩比相同时的 attention 输出误差：INT4 是 0.007（KIVI）、
0.052（QuaRot）、0.064（RTN）；INT2 是 0.107（KIVI）、0.416（QuaRot）、0.532（RTN）。

## 生成协议

prompt、种子、注意力窗口、帧数都和 Tempokv、QVG 的 runner 对齐，三个仓库的视频可以直接
比较：MovieGen-128 prompt、噪声种子 `seed + prompt_index * 1000003`、全历史注意力、
180 个 latent 帧（16fps 下 717 帧）。

## 目录

```text
kv_quant/                     三个模型共用的量化器
Self-Forcing/scripts/         生成、评测、汇总
Causal-Forcing/               集成、inference.py、批量脚本
LongCat/                      集成、run_long_t2v.py、批量脚本
scripts/paper_experiments/    efficiency 与 latency
third_party/Quant-VideoGen/   QVG 官方代码
```

## 安装

```bash
pip install -r Self-Forcing/requirements-inference.txt   # 生成
pip install -r Self-Forcing/requirements-eval.txt        # VBench 与保真度
pip install -r Causal-Forcing/requirements.txt           # Causal-Forcing
pip install -r Causal-Forcing/requirements-qvg.txt       # 仅 QVG
```

## Self-Forcing

```bash
python Self-Forcing/scripts/01_generate.py \
  --method RTN_INT4 \
  --checkpoint-path /path/to/self_forcing_dmd.pt \
  --results-root results
```

在已有 BF16 run 的基础上跑完六行 INT4 / INT2：

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

批量：`CONFIG_PATH=... CHECKPOINT_PATH=... DATA_PATH=... bash
Causal-Forcing/run_baseline_matrix.sh`。QVG 用单独的 `run_qvg.sh`，需要
`requirements-qvg.txt`。

## LongCat

所有续写都从同一个初始视频开始：

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

行与行之间只改 `--method`。批量脚本是 `LongCat/run_baseline_matrix.sh`。

## 指标

每次运行都会写出运行时间、峰值显存和 cache 字节：Self-Forcing 写
`metrics/efficiency_<METHOD>.json`，另外两个模型每次运行写一份 report。压缩比用
`resident_analytic.v1` 口径：分子是当前真正持有的 cache 位置换成 BF16 的字节数，分母是
同一时刻的常驻字节，不是预分配的容量。

评测：`02_eval_fidelity.py`（对 BF16 的 PSNR/SSIM/LPIPS）、`03_eval_vbench.sh`、
`05_summarize_results.py`。
