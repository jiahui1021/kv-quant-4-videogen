import os
import random
import argparse
import datetime
import json
import sys
import time
from pathlib import Path
import PIL.Image
import numpy as np

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
LONGCAT_ROOT = Path(__file__).resolve().parent
for _path in (REPO_ROOT, LONGCAT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video, read_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel

from prompt_loader import load_prompt_or_image

from quant_videogen.timer import print_operator_log_data

from quant_videogen.misc import Color
from quant_videogen.logger import logger
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
from kv_quant.factory import SUPPORTED_METHODS, parse_method


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed_environment():

    # prepare distributed environment
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24)
    )
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    return global_rank, num_processes, local_rank


def init_cp(context_parallel_size, global_rank, num_processes):
    # initialize context parallel before loading models
    init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes,
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    return cp_split_hw


def _distributed_max_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _write_generation_metrics(args, dit, device, start_time: float, prompt_idx: int, seed: int):
    """Write one machine-readable report for the completed video generation."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    wall_clock_runtime_s = time.perf_counter() - start_time
    peak_vram_bytes = _distributed_max_int(
        torch.cuda.max_memory_allocated(device), device
    )

    events = list(getattr(dit, "kv_quant_events", []))
    peak_bf16_kv_bytes = max(
        (int(event.get("bf16_bytes", 0)) for event in events),
        default=0,
    )
    peak_compressed_kv_bytes = max(
        (int(event.get("compressed_bytes", 0)) for event in events),
        default=0,
    )
    quantizer = getattr(dit, "kv_quantizer", None)
    stats = getattr(quantizer, "stats", None)
    if stats is not None and hasattr(stats, "resolve_timing"):
        stats.resolve_timing(synchronize=False)
    method = getattr(dit, "kv_quant_method", "BF16")
    if method == "BF16" and peak_bf16_kv_bytes:
        peak_compressed_kv_bytes = peak_bf16_kv_bytes

    report = {
        "model": "longcat",
        "method": method,
        "block_size": int(args.block_size),
        "bits": None if method == "BF16" else int(method.rsplit("INT", 1)[1]),
        "end_to_end_generation_time_s": float(wall_clock_runtime_s),
        "wall_clock_runtime_s": float(wall_clock_runtime_s),
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_gb": peak_vram_bytes / 1024**3,
        "peak_bf16_kv_bytes": peak_bf16_kv_bytes,
        "peak_compressed_kv_bytes": peak_compressed_kv_bytes,
        "compression_ratio": (
            peak_bf16_kv_bytes / peak_compressed_kv_bytes
            if peak_compressed_kv_bytes
            else 0.0
        ),
        "quantize_time_s": 0.0 if stats is None else float(stats.quantize_time_s),
        "dequantize_time_s": 0.0 if stats is None else float(stats.dequantize_time_s),
        "quantize_calls": 0 if stats is None else int(stats.quantize_calls),
        "dequantize_calls": 0 if stats is None else int(stats.dequantize_calls),
        "num_kv_segments": len(events),
        "prompt_idx": int(prompt_idx),
        "seed": int(seed),
    }

    if quantizer is not None and stats.quantize_calls <= 0:
        raise RuntimeError("KV quantization was never triggered.")
    if quantizer is not None and stats.dequantize_calls <= 0:
        raise RuntimeError("KV dequantization was never triggered.")

    metrics_dir = Path(args.output_dir) / f"{prompt_idx}-{seed}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"metrics_{method}.json"
    if dist.is_initialized():
        dist.barrier()
    if int(os.environ.get("RANK", "0")) == 0:
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics written to {metrics_path}")


def get_model_and_pipe(
    checkpoint_dir, local_rank, global_rank, num_processes, cp_split_hw, enable_compile
):

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir,
        subfolder="dit",
        cp_split_hw=cp_split_hw,
        torch_dtype=torch.bfloat16,
    )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    if enable_compile:
        for layer_idx in range(len(pipe.dit.blocks)):
            pipe.dit.blocks[layer_idx] = torch.compile(pipe.dit.blocks[layer_idx])
        # pipe.dit = torch.compile(pipe.dit)

    for layer in pipe.dit.blocks:
        layer.register_forward_hook(print_operator_log_data)

    return dit, pipe


def calculate_frame_parameters(pipe, num_frames, num_cond_frames, video):
    q_num_frame = (
        num_frames - num_cond_frames - 1
    ) // pipe.vae_scale_factor_temporal + 1
    kv_num_frame = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1

    height, width = video[0].size
    frame_size = height * width / (pipe.vae_scale_factor_spatial * 2) ** 2

    return int(q_num_frame), int(kv_num_frame), int(frame_size)


def generate(args):
    # case setup - load prompt from args
    prompt_source = args.prompt_source
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    prompt_idx = args.prompt_idx
    loaded = load_prompt_or_image(prompt_source, prompt_idx, prompt)
    if isinstance(loaded, tuple):
        prompt, image_path = loaded
    else:
        prompt = loaded

    seed = args.seed

    seed_everything(seed)

    # load generation parameters from args
    num_segments = args.num_segments
    num_frames = args.num_frames
    num_cond_frames = args.num_cond_frames
    spatial_refine_only = args.spatial_refine_only

    # load parsed args
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile

    global_rank, num_processes, local_rank = init_distributed_environment()
    cp_split_hw = init_cp(context_parallel_size, global_rank, num_processes)
    dit, pipe = get_model_and_pipe(
        checkpoint_dir,
        local_rank,
        global_rank,
        num_processes,
        cp_split_hw,
        enable_compile,
    )
    method_name, quantizer = parse_method(args.method, block_size=args.block_size)
    if quantizer is not None:
        quantizer.set_timing_enabled(args.profile_quant_timing)
    if method_name != "BF16" and args.quant_type != "none":
        raise ValueError(
            "Do not enable shared RTN/KIVI/QuaRot and legacy --quant_type at the same time."
        )
    dit.set_kv_quantizer(method_name, quantizer)
    dit.reset_kv_quant_metrics()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(local_rank)
    generation_start_time = time.perf_counter()
    logger.info(f"{Color.green}Prompt: {prompt}{Color.reset}")

    logger.info(
        f"{Color.green}Memory usage after loading model: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB{Color.reset}"
    )

    ##################################################################################################################
    # Text-to-Video (480p) - Initial Video Generation
    ##################################################################################################################
    if args.workload == "480p_init":
        if local_rank == 0:
            print("================ Generating initial video (480p)...")
        output = pipe.generate_t2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=num_frames,
            num_inference_steps=50,
            guidance_scale=4.0,
        )[0]

        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(output))
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            save_path = os.path.join(args.output_dir, f"{prompt_idx}-{seed}.mp4")

            print(f"Saving video to {save_path}")

            write_video(
                save_path,
                output_tensor,
                fps=15,
                video_codec="libx264",
                options={"crf": f"{18}"},
            )

        del output
        torch_gc()

    ##################################################################################################################
    # Low Resolution Generation - Long Video Extension
    ##################################################################################################################
    if args.workload == "480p_long_gen":
        # Load initial video from path
        if args.init_video_path is None:
            raise ValueError("--init_video_path is required for low_res_gen workload")

        if local_rank == 0:
            print(
                f"================ Loading initial video from {args.init_video_path}..."
            )
        output = read_video(args.init_video_path)[0]

        video = [(output[i].numpy()).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]
        del output
        torch_gc()

        target_size = video[0].size
        current_video = video
        all_generated_frames = video

        num_generate_frames = num_frames - num_cond_frames

        if args.attn_sink_frames > 0:
            sink_video = video[: args.attn_sink_frames]
            print(f"Sinking {len(sink_video)} frames")

            current_video = sink_video + current_video[-num_cond_frames:]

            new_num_cond_frames = num_cond_frames + args.attn_sink_frames
            new_num_frames = num_frames + args.attn_sink_frames
        else:
            current_video = current_video[-num_cond_frames:]

            new_num_cond_frames = num_cond_frames
            new_num_frames = num_frames

        # Resigter Quantization Related Configurations
        pipe.dit.quant_config = QuantizeConfig(
            quant_type=args.quant_type,
            quant_block_size=args.quant_block_size,
            num_prq_stages=args.num_prq_stages,
            cache_num_k_centroids=args.cache_num_k_centroids,
            cache_num_v_centroids=args.cache_num_v_centroids,
            kmeans_max_iters=args.kmeans_max_iters,
        )

        for segment_idx in range(num_segments):
            if local_rank == 0:
                print(
                    f"================ Generating segment {segment_idx + 1}/{num_segments}..."
                )

            # Pass the number of segments by environment variable
            os.environ["SEGMENT_IDX"] = str(segment_idx)

            print(
                f"Current video size: {len(current_video)} Frame x {current_video[0].size} Pixel. Condition Frame: {new_num_cond_frames}. Generate Frame: {num_generate_frames}. Total Frame: {new_num_frames}."
            )

            output = pipe.generate_vc(
                video=current_video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution="480p",  # 480p / 720p
                num_frames=new_num_frames,
                num_cond_frames=new_num_cond_frames,
                num_inference_steps=50,
                guidance_scale=4.0,
                use_kv_cache=True,
                offload_kv_cache=args.offload_kv_cache,
                enhance_hf=True,
            )[0]

            print(f"Output shape: {output.shape}")

            new_video = [
                (output[i] * 255).astype(np.uint8) for i in range(output.shape[0])
            ]
            new_video = [PIL.Image.fromarray(img) for img in new_video]
            new_video = [
                frame.resize(target_size, PIL.Image.BICUBIC) for frame in new_video
            ]
            del output

            all_generated_frames.extend(new_video[new_num_cond_frames:])

            current_video = new_video

            if local_rank == 0:
                output_tensor = torch.from_numpy(np.array(all_generated_frames))
                save_path = os.path.join(
                    args.output_dir,
                    f"{prompt_idx}-{seed}",
                    f"segment_{segment_idx+1}.mp4",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"Saving video to {save_path}")

                write_video(
                    save_path,
                    output_tensor,
                    fps=15,
                    video_codec="libx264",
                    options={"crf": f"{18}"},
                )
                del output_tensor

    ##################################################################################################################
    # Low Resolution Generation - Long Video Extension
    ##################################################################################################################
    if args.workload == "480p_long_gen_fullkv":
        # Load initial video from path
        if args.init_video_path is None:
            raise ValueError(
                "--init_video_path is required for low_res_gen_fullkv workload"
            )

        if local_rank == 0:
            print(
                f"================ Loading initial video from {args.init_video_path}..."
            )
        output = read_video(args.init_video_path)[0]

        video = [(output[i].numpy()).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]
        del output
        torch_gc()

        target_size = video[0].size
        current_video = video
        all_generated_frames = video

        num_generate_frames = num_frames - num_cond_frames

        # Resigter Quantization Related Configurations
        pipe.dit.quant_config = QuantizeConfig(
            quant_type=args.quant_type,
            quant_block_size=args.quant_block_size,
            num_prq_stages=args.num_prq_stages,
            cache_num_k_centroids=args.cache_num_k_centroids,
            cache_num_v_centroids=args.cache_num_v_centroids,
            kmeans_max_iters=args.kmeans_max_iters,
        )

        for segment_idx in range(num_segments):
            if local_rank == 0:
                print(
                    f"================ Generating segment {segment_idx + 1}/{num_segments}..."
                )

            # Pass the number of segments by environment variable
            os.environ["SEGMENT_IDX"] = str(segment_idx)

            new_num_cond_frames = len(current_video)
            new_num_frames = new_num_cond_frames + num_generate_frames

            print(
                f"Current video size: {len(current_video)} Frame x {current_video[0].size} Pixel. Condition Frame: {new_num_cond_frames}. Generate Frame: {num_generate_frames}. Total Frame: {new_num_frames}."
            )

            output = pipe.generate_vc(
                video=current_video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution="480p",  # 480p / 720p
                num_frames=new_num_frames,
                num_cond_frames=new_num_cond_frames,
                num_inference_steps=50,
                guidance_scale=4.0,
                use_kv_cache=True,
                offload_kv_cache=args.offload_kv_cache,
                enhance_hf=True,
            )[0]

            print(f"Output shape: {output.shape}")

            new_video = [
                (output[i] * 255).astype(np.uint8) for i in range(output.shape[0])
            ]
            new_video = [PIL.Image.fromarray(img) for img in new_video]
            new_video = [
                frame.resize(target_size, PIL.Image.BICUBIC) for frame in new_video
            ]
            del output

            # I change the following lines!
            all_generated_frames.extend(new_video[new_num_cond_frames:])

            current_video = new_video

            if local_rank == 0:
                output_tensor = torch.from_numpy(np.array(all_generated_frames))
                save_path = os.path.join(
                    args.output_dir,
                    f"{prompt_idx}-{seed}",
                    f"segment_{segment_idx+1}.mp4",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"Saving video to {save_path}")

                write_video(
                    save_path,
                    output_tensor,
                    fps=15,
                    video_codec="libx264",
                    options={"crf": f"{18}"},
                )
                del output_tensor

    ##################################################################################################################
    # High Resolution Refinement (720p)
    ##################################################################################################################
    if args.workload == "704p_long_refine":
        # Load video for refinement from path
        if args.refine_video_path is None:
            raise ValueError(
                "--refine_video_path is required for high_res_refine workload"
            )

        if local_rank == 0:
            print(
                f"================ Loading video for refinement from {args.refine_video_path}..."
            )

        output_for_refine = read_video(args.refine_video_path)[0]

        all_generated_frames = [
            (output_for_refine[i].numpy()).astype(np.uint8)
            for i in range(output_for_refine.shape[0])
        ]
        all_generated_frames = [
            PIL.Image.fromarray(img) for img in all_generated_frames
        ]

        refinement_lora_path = os.path.join(
            checkpoint_dir, "lora/refinement_lora.safetensors"
        )
        pipe.dit.load_lora(refinement_lora_path, "refinement_lora")
        pipe.dit.enable_loras(["refinement_lora"])

        if enable_compile:
            for layers in pipe.dit.blocks:
                layers = torch.compile(layers)
            # dit = torch.compile(dit)

        torch_gc()
        cur_condition_video = None
        cur_num_cond_frames = 0
        start_id = 0
        all_refine_frames = []

        for segment_idx in range(num_segments + 1):
            if local_rank == 0:
                print(f"Refine segment {segment_idx + 1}/{num_segments+1}...")

            output_refine = pipe.generate_refine(
                video=cur_condition_video,
                prompt="",
                stage1_video=all_generated_frames[start_id : start_id + num_frames],
                num_cond_frames=cur_num_cond_frames,
                num_inference_steps=50,
                spatial_refine_only=spatial_refine_only,
            )[0]

            new_video = [
                (output_refine[i] * 255).astype(np.uint8)
                for i in range(output_refine.shape[0])
            ]
            new_video = [PIL.Image.fromarray(img) for img in new_video]
            del output_refine

            all_refine_frames.extend(new_video[cur_num_cond_frames:])
            cur_condition_video = new_video
            cur_num_cond_frames = (
                num_cond_frames if spatial_refine_only else num_cond_frames * 2
            )
            start_id = start_id + num_frames - num_cond_frames

            if local_rank == 0:
                output_tensor = torch.from_numpy(np.array(all_refine_frames))
                fps = 15 if spatial_refine_only else 30
                save_path = os.path.join(
                    args.output_dir,
                    f"{prompt_idx}-{seed}",
                    f"refine_{segment_idx}.mp4",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"Saving video to {save_path}")

                write_video(
                    save_path,
                    output_tensor,
                    fps=fps,
                    video_codec="libx264",
                    options={"crf": f"{10}"},
                )

    ##################################################################################################################
    # High Resolution Refinement (720p) with BSA
    ##################################################################################################################
    if args.workload == "704p_long_refine_bsa":
        # Load video for refinement from path
        if args.refine_video_path is None:
            raise ValueError(
                "--refine_video_path is required for high_res_refine workload"
            )

        if local_rank == 0:
            print(
                f"================ Loading video for refinement from {args.refine_video_path}..."
            )

        output_for_refine = read_video(args.refine_video_path)[0]

        all_generated_frames = [
            (output_for_refine[i].numpy()).astype(np.uint8)
            for i in range(output_for_refine.shape[0])
        ]
        all_generated_frames = [
            PIL.Image.fromarray(img) for img in all_generated_frames
        ]
        del output_for_refine

        refinement_lora_path = os.path.join(
            checkpoint_dir, "lora/refinement_lora.safetensors"
        )
        pipe.dit.load_lora(refinement_lora_path, "refinement_lora")
        pipe.dit.enable_loras(["refinement_lora"])
        pipe.dit.enable_bsa()

        if enable_compile:
            for layers in pipe.dit.blocks:
                layers = torch.compile(layers)
            # dit = torch.compile(dit)

        torch_gc()
        cur_condition_video = None
        cur_num_cond_frames = 0
        start_id = 0
        all_refine_frames = []

        for segment_idx in range(num_segments + 1):
            if local_rank == 0:
                print(f"Refine segment {segment_idx + 1}/{num_segments+1}...")

            output_refine = pipe.generate_refine(
                video=cur_condition_video,
                prompt="",
                stage1_video=all_generated_frames[start_id : start_id + num_frames],
                num_cond_frames=cur_num_cond_frames,
                num_inference_steps=50,
                spatial_refine_only=spatial_refine_only,
            )[0]

            new_video = [
                (output_refine[i] * 255).astype(np.uint8)
                for i in range(output_refine.shape[0])
            ]
            new_video = [PIL.Image.fromarray(img) for img in new_video]
            del output_refine

            all_refine_frames.extend(new_video[cur_num_cond_frames:])
            cur_condition_video = new_video
            cur_num_cond_frames = (
                num_cond_frames if spatial_refine_only else num_cond_frames * 2
            )
            start_id = start_id + num_frames - num_cond_frames

            if local_rank == 0:
                output_tensor = torch.from_numpy(np.array(all_refine_frames))
                fps = 15 if spatial_refine_only else 30
                save_path = os.path.join(
                    args.output_dir,
                    f"{prompt_idx}-{seed}",
                    f"refine_{segment_idx}.mp4",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"Saving video to {save_path}")

                write_video(
                    save_path,
                    output_tensor,
                    fps=fps,
                    video_codec="libx264",
                    options={"crf": f"{10}"},
                )

    _write_generation_metrics(
        args,
        dit,
        torch.device(f"cuda:{local_rank}"),
        generation_start_time,
        prompt_idx,
        seed,
    )


def _parse_args():
    parser = argparse.ArgumentParser()

    # Model Configuration Arguments
    model_group = parser.add_argument_group(
        "Model Configuration", "Arguments for model setup and optimization"
    )
    model_group.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Size of context parallelism",
    )
    model_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to model checkpoint directory",
    )
    model_group.add_argument(
        "--enable_compile",
        action="store_true",
        help="Enable torch.compile for model optimization",
    )

    # Prompt Configuration Arguments
    prompt_group = parser.add_argument_group(
        "Prompt Configuration", "Arguments for prompt configuration"
    )
    prompt_group.add_argument(
        "--prompt_source",
        type=str,
        default="direct_prompt",
        choices=[
            "direct_prompt",
            "image_to_video_vbench",
            "image_to_video_from_json",
            "text_to_video_from_file",
            "text_to_video_from_json",
        ],
        help="Source of prompt for video generation",
    )
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default="realistic filming style, a person wearing a dark helmet, a deep-colored jacket, blue jeans, and bright yellow shoes rides a skateboard along a winding mountain road. The skateboarder starts in a standing position, then gradually lowers into a crouch, extending one hand to touch the road surface while maintaining a low center of gravity to navigate a sharp curve. After completing the turn, the skateboarder rises back to a standing position and continues gliding forward. The background features lush green hills flanking both sides of the road, with distant snow-capped mountain peaks rising against a clear, bright blue sky. The camera follows closely from behind, smoothly tracking the skateboarder's movements and capturing the dynamic scenery along the route. The scene is shot in natural daylight, highlighting the vivid outdoor environment and the skateboarder's fluid actions.",
        help="Text prompt for video generation",
    )
    prompt_group.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Negative text prompt for video generation",
    )
    prompt_group.add_argument(
        "--prompt_idx",
        type=int,
        default=0,
        help="Index of prompt to use from prompt file",
    )

    # Generation Parameters Arguments
    gen_group = parser.add_argument_group(
        "Generation Parameters", "Arguments for video generation settings"
    )
    gen_group.add_argument(
        "--num_segments",
        type=int,
        default=8,
        help="Number of segments for long video generation (8 segments = 1 minute video)",
    )
    gen_group.add_argument(
        "--num_frames",
        type=int,
        default=93,
        help="Number of frames per segment",
    )
    gen_group.add_argument(
        "--num_cond_frames",
        type=int,
        default=53,
        help="Number of conditioning frames for video continuation",
    )
    gen_group.add_argument(
        "--attn_sink_frames",
        type=int,
        default=0,
        help="Number of frames to sink for attention",
    )
    gen_group.add_argument(
        "--spatial_refine_only",
        action="store_true",
        help="Enable spatial refinement only (no temporal upsampling)",
    )
    gen_group.add_argument(
        "--offload_kv_cache",
        action="store_true",
        default=True,
        help="Offload KV cache to CPU to save VRAM (default: True)",
    )
    gen_group.add_argument(
        "--no_offload_kv_cache",
        action="store_false",
        dest="offload_kv_cache",
        help="Disable KV cache offload (keep on GPU)",
    )

    # Workload Configuration Arguments
    workload_group = parser.add_argument_group(
        "Workload Configuration", "Arguments for workload and I/O paths"
    )
    workload_group.add_argument(
        "--workload",
        type=str,
        default="480p_init",
        choices=[
            "480p_init",
            "480p_long_gen",
            "480p_long_gen_fullkv",
            "704p_long_refine",
            "704p_long_refine_bsa",
        ],
        help="Type of workload to run",
    )
    workload_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for workload",
    )
    workload_group.add_argument(
        "--output_dir",
        type=str,
        default="results/long_t2v",
        help="Directory for output videos",
    )
    workload_group.add_argument(
        "--refine_video_path",
        type=str,
        default=None,
        help="Path to video for refinement (required for refine workloads)",
    )
    workload_group.add_argument(
        "--init_video_path",
        type=str,
        default=None,
        help="Path to initial video (required for long generation workloads)",
    )

    # Quantization Arguments
    quant_group = parser.add_argument_group(
        "Quantization", "Arguments for model quantization"
    )
    quant_group.add_argument(
        "--method",
        type=str,
        default="BF16",
        # QVG is a Causal-Forcing-only backend; LongCat must not advertise it.
        choices=[
            method for method in SUPPORTED_METHODS if not method.startswith("QVG_")
        ],
        help="Shared KV-cache method used by LongCat generation",
    )
    quant_group.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Sequence block size for shared RTN/KIVI/QuaRot KV quantization",
    )
    quant_group.add_argument(
        "--profile_quant_timing",
        action="store_true",
        help="Record optional quantize/dequantize CUDA-event breakdowns",
    )
    quant_group.add_argument(
        "--quant_type",
        type=str,
        default="none",
        choices=[
            "none",
            "naive-fp4",
            "kmeans-fp4",
            "kmeans-fp4-clip",
            "nstages-kmeans-fp4",
            "nstages-kmeans-fp4-clip",
            "naive-int4",
            "kmeans-int4",
            "kmeans-int4-clip",
            "nstages-kmeans-int4",
            "nstages-kmeans-int4-clip",
            "naive-int3",
            "kmeans-int3",
            "kmeans-int3-clip",
            "nstages-kmeans-int3",
            "nstages-kmeans-int3-clip",
            "naive-int2",
            "kmeans-int2",
            "kmeans-int2-clip",
            "nstages-kmeans-int2",
            "nstages-kmeans-int2-clip",
            "triton-nstages-kmeans-int2",
            "triton-nstages-kmeans-int2-clip",
            "triton-nstages-kmeans-int4",
            "triton-nstages-kmeans-int4-clip",
        ],
        help="Quantization type for the model",
    )

    quant_group.add_argument(
        "--quant_block_size",
        type=int,
        default=16,
        help="Block size for quantization",
    )

    quant_group.add_argument(
        "--num_prq_stages",
        type=int,
        default=4,
        help="Number of prq stages for nstages-kmeans quantization",
    )

    quant_group.add_argument(
        "--cache_num_k_centroids",
        type=int,
        default=256,
        help="Number of K-Means centroids for K tensor (used in kmeans and nstages-kmeans)",
    )

    quant_group.add_argument(
        "--cache_num_v_centroids",
        type=int,
        default=256,
        help="Number of K-Means centroids for V tensor (used in kmeans and nstages-kmeans)",
    )

    quant_group.add_argument(
        "--kmeans_max_iters",
        type=int,
        default=100,
        help="Maximum iterations for K-Means clustering",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate(args)
