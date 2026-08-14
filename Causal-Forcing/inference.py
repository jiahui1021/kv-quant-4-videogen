import argparse
import json
import sys
import time
from pathlib import Path
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
CAUSAL_ROOT = Path(__file__).resolve().parent
for _path in (REPO_ROOT, CAUSAL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from kv_quant.factory import SUPPORTED_METHODS, parse_method
from kv_quant_runtime import (
    attach_quantizer_to_pipeline,
    finalize_quantized_kv_cache,
    reset_quantized_kv_cache,
    resident_kv_memory_bytes,
)
from qvg_runtime import (
    QVGConfig,
    attach_qvg_to_pipeline,
    qvg_metrics,
    qvg_memory_breakdown,
    qvg_resident_memory_bytes,
    reset_qvg_cache,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=180, help="Number of latent frames to generate (180 = 717 pixel frames)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--report_timing", action="store_true",
                    help="Only tested on A800, for the Causal Forcing++ latency. Not make claims for other hardware like H100. For the result on H100, refer to the reported results in the Self Forcing paper.")
parser.add_argument(
    "--method",
    type=str,
    default="BF16",
    choices=list(SUPPORTED_METHODS),
    help="Shared causal self-attention KV-cache method",
)
parser.add_argument(
    "--block_size",
    type=int,
    default=16,
    help="Sequence block size for shared RTN/KIVI/QuaRot KV quantization",
)
parser.add_argument(
    "--profile_quant_timing",
    action="store_true",
    help="Record optional quantize/dequantize CUDA-event breakdowns",
)
parser.add_argument("--qvg_quant_factor", type=int, default=8)
parser.add_argument("--qvg_num_k_centroids", type=int, default=256)
parser.add_argument("--qvg_num_v_centroids", type=int, default=256)
parser.add_argument("--qvg_kmeans_max_iters", type=int, default=2)
parser.add_argument("--qvg_quant_block_size", type=int, default=64)
parser.add_argument("--qvg_num_prq_stages", type=int, default=1)
parser.add_argument(
    "--qvg_disable_compression",
    action="store_true",
    help="Keep QVG cache entries in BF16 for cache/RoPE equivalence debugging",
)
parser.add_argument(
    "--local_attn_size",
    type=int,
    default=None,
    help="Override the causal model attention window in latent frames",
)
parser.add_argument(
    "--retain_final_cache",
    action="store_true",
    help="run the final clean refresh before releasing the cache",
)
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load(str(CAUSAL_ROOT / "configs/default_config.yaml"))
config = OmegaConf.merge(default_config, config)
if args.local_attn_size is not None:
    if not hasattr(config, "model_kwargs") or config.model_kwargs is None:
        config.model_kwargs = OmegaConf.create({})
    config.model_kwargs.local_attn_size = int(args.local_attn_size)

num_frame_per_block = int(getattr(config, "num_frame_per_block", 1))
latent_frames = int(args.num_output_frames)
if latent_frames % num_frame_per_block:
    raise ValueError(
        "num_output_frames must be divisible by num_frame_per_block for the "
        "formal Causal-Forcing workload"
    )
qvg_enabled = args.method.startswith("QVG_")
effective_local_attn_size = int(
    args.local_attn_size
    if args.local_attn_size is not None
    else getattr(config.model_kwargs, "local_attn_size", -1)
)
benchmark_config = {
    "method": args.method,
    "pixel_frames": latent_frames * 4 - 3,
    "latent_frames": latent_frames,
    "num_frame_per_block": num_frame_per_block,
    "num_blocks": latent_frames // num_frame_per_block,
    "local_attn_size": effective_local_attn_size,
    "attention_mode": (
        "full_history"
        if effective_local_attn_size in (-1, latent_frames)
        else "windowed"
    ),
    "compression_span_frames": (
        int(args.qvg_quant_factor * num_frame_per_block)
        if qvg_enabled
        else None
    ),
    "seed": int(args.seed),
    "checkpoint": str(Path(args.checkpoint_path).expanduser().resolve()),
    "prompt_file": str(Path(args.data_path).expanduser().resolve()),
}
if qvg_enabled:
    benchmark_config.update(
        {
            "qvg_quant_factor": int(args.qvg_quant_factor),
            "qvg_num_k_centroids": int(args.qvg_num_k_centroids),
            "qvg_num_v_centroids": int(args.qvg_num_v_centroids),
            "qvg_kmeans_max_iters": int(args.qvg_kmeans_max_iters),
            "qvg_quant_block_size": int(args.qvg_quant_block_size),
            "qvg_num_prq_stages": int(args.qvg_num_prq_stages),
        }
    )
print("[BenchmarkConfig]", flush=True)
for key, value in benchmark_config.items():
    print(f"{key} = {value}", flush=True)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator_ema' if args.use_ema else 'generator'
    gen_sd = state_dict[key]

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)

if qvg_enabled:
    if args.i2v:
        raise NotImplementedError(
            "QVG Causal-Forcing I2V path is not yet validated; use T2V first."
        )
    if isinstance(pipeline, CausalDiffusionInferencePipeline):
        raise NotImplementedError(
            "QVG backend is not yet validated for "
            "CausalDiffusionInferencePipeline"
        )
    method_name = args.method
    quantizer = None
    qvg_disable_compression = args.qvg_disable_compression or (
        os.environ.get("QVG_BF16_DEBUG", "").lower()
        in {"1", "true", "yes", "on"}
    )
    qvg_config = QVGConfig.from_method(
        method_name,
        num_k_centroids=args.qvg_num_k_centroids,
        num_v_centroids=args.qvg_num_v_centroids,
        kmeans_max_iters=args.qvg_kmeans_max_iters,
        quant_block_size=args.qvg_quant_block_size,
        num_prq_stages=args.qvg_num_prq_stages,
        quant_factor=args.qvg_quant_factor,
        timing_enabled=args.profile_quant_timing,
        compression_enabled=not qvg_disable_compression,
    )
    attach_qvg_to_pipeline(
        pipeline,
        qvg_config,
        num_output_frames=args.num_output_frames,
        dtype=torch.bfloat16,
        device=device,
    )
else:
    method_name, quantizer = parse_method(
        args.method, block_size=args.block_size
    )

if quantizer is not None:
    quantizer.set_timing_enabled(args.profile_quant_timing)
    attach_quantizer_to_pipeline(
        pipeline,
        quantizer,
        num_output_frames=args.num_output_frames,
        dtype=torch.bfloat16,
        device=device,
    )


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def _distributed_max_memory(device: torch.device) -> int:
    value = torch.tensor(
        [torch.cuda.max_memory_allocated(device)], device=device, dtype=torch.long
    )
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return int(value.item())


def _write_metrics(
    prompt_idx: int,
    method_name: str,
    quantizer,
    pipeline,
    device: torch.device,
    start_time: float,
    output_folder: str,
    block_size: int,
    benchmark_config: dict[str, object],
) -> None:
    torch.cuda.synchronize(device)
    is_qvg = method_name.startswith("QVG_")
    if not is_qvg:
        finalize_quantized_kv_cache(pipeline, quantizer)
    elapsed = time.perf_counter() - start_time
    peak_vram_bytes = _distributed_max_memory(device)
    if is_qvg:
        bf16_kv_bytes, compressed_kv_bytes = qvg_resident_memory_bytes(
            pipeline
        )
        stats = pipeline.qvg_stats
    else:
        bf16_kv_bytes, compressed_kv_bytes = resident_kv_memory_bytes(
            pipeline, quantizer
        )
        stats = getattr(quantizer, "stats", None)
    if stats is not None and hasattr(stats, "resolve_timing"):
        stats.resolve_timing(synchronize=False)
    report = {
        "model": "causal_forcing",
        "method": method_name,
        "block_size": None if is_qvg else int(block_size),
        "bits": (
            int(pipeline.qvg_config.bits)
            if is_qvg
            else None if method_name == "BF16" else int(quantizer.bits)
        ),
        "end_to_end_generation_time_s": float(elapsed),
        "wall_clock_runtime_s": float(elapsed),
        "diffusion_generation_s": (
            None
            if getattr(pipeline, "diffusion_generation_s", None) is None
            else float(pipeline.diffusion_generation_s)
        ),
        "vae_decode_s": (
            None
            if getattr(pipeline, "vae_decode_s", None) is None
            else float(pipeline.vae_decode_s)
        ),
        "benchmark_config": benchmark_config,
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_gb": peak_vram_bytes / 1024**3,
        "resident_bf16_kv_bytes": int(bf16_kv_bytes),
        "resident_compressed_kv_bytes": int(compressed_kv_bytes),
        "resident_logical_kv_values": int(bf16_kv_bytes // 2),
        # Keep the previous keys for existing result readers. Their values now
        # use resident-capacity accounting as well.
        "active_bf16_kv_bytes": int(bf16_kv_bytes),
        "active_compressed_kv_bytes": int(compressed_kv_bytes),
        "effective_kv_bits_per_value": (
            compressed_kv_bytes * 8 / max(bf16_kv_bytes / 2, 1)
            if compressed_kv_bytes
            else 16.0
        ),
        "compression_ratio": (
            bf16_kv_bytes / compressed_kv_bytes if compressed_kv_bytes else 0.0
        ),
        "quantize_time_s": 0.0 if stats is None else float(stats.quantize_time_s),
        "dequantize_time_s": 0.0 if stats is None else float(stats.dequantize_time_s),
        "quantize_calls": 0 if stats is None else int(stats.quantize_calls),
        "dequantize_calls": 0 if stats is None else int(stats.dequantize_calls),
        "prompt_idx": int(prompt_idx),
    }
    if is_qvg:
        qvg_memory = qvg_memory_breakdown(pipeline)
        report.update(
            {
                "resident_bf16_kv_bytes": int(
                    qvg_memory.physical_bf16_bytes
                ),
                "resident_compressed_kv_bytes": int(
                    qvg_memory.physical_bytes
                ),
                "active_bf16_kv_bytes": int(qvg_memory.physical_bf16_bytes),
                "active_compressed_kv_bytes": int(
                    qvg_memory.physical_bytes
                ),
                "qvg_packed_bytes": int(
                    qvg_memory.physical_compressed_bytes
                ),
                "qvg_bf16_tail_bytes": int(qvg_memory.physical_bf16_bytes),
                "resident_total_kv_bytes": int(qvg_memory.physical_bytes),
                "uncompressed_reference_kv_bytes": int(
                    qvg_memory.bf16_equivalent_bytes
                ),
                "resident_logical_kv_values": int(qvg_memory.logical_values),
                "effective_kv_bits_per_value": float(
                    qvg_memory.physical_bytes
                    * 8
                    / max(qvg_memory.logical_values, 1)
                ),
                "compression_ratio": float(
                    qvg_memory.bf16_equivalent_bytes
                    / max(qvg_memory.physical_bytes, 1)
                ),
            }
        )
        report.update(qvg_metrics(pipeline))
    qvg_compression_enabled = (
        is_qvg and bool(pipeline.qvg_config.compression_enabled)
    )
    if quantizer is not None or qvg_compression_enabled:
        if stats.quantize_calls <= 0:
            raise RuntimeError("KV quantization was never triggered.")
        if stats.dequantize_calls <= 0:
            raise RuntimeError("KV dequantization was never triggered.")

    metrics_dir = Path(output_folder)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"metrics_{method_name}_{prompt_idx}.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if local_rank == 0:
        print(f"Metrics written to {metrics_path}")


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    
    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt] 
        sampled_noise = torch.randn(
            [1, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    if qvg_enabled:
        reset_qvg_cache(pipeline)
    elif quantizer is not None:
        reset_quantized_kv_cache(pipeline)
        quantizer.reset_stats()
    torch.cuda.reset_peak_memory_stats(device)
    generation_start_time = time.perf_counter()

    sample_report_timing = args.report_timing
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        report_timing=sample_report_timing,
        retain_final_cache=args.retain_final_cache,
    )
    if sample_report_timing:
        latency = pipeline.first_chunk_time
        elapsed = pipeline.last_generation_time
        num_pixel_frames = video.shape[1]
        fps = num_pixel_frames / elapsed if elapsed > 0 else float('inf')
        print(f"[Sample {i}] {num_pixel_frames} frames, "
              f"latency ↓ {latency:.2f}s, FPS ↑ {fps:.2f}")
        # Only tested on A800, for the Causal Forcing++ paper latency & throughput.
        # Not make claims for other hardware like H100.
        # For the result on H100, refer to the reported results in the Self Forcing paper.
        # We do not guarantee that our FPS/latency measurement protocol is identical to that used in the Self Forcing paper.
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
    write_video(output_path, video[0], fps=16)
    _write_metrics(
        prompt_idx=idx,
        method_name=method_name,
        quantizer=quantizer,
        pipeline=pipeline,
        device=device,
        start_time=generation_start_time,
        output_folder=args.output_folder,
        block_size=args.block_size,
        benchmark_config=benchmark_config,
    )

       
