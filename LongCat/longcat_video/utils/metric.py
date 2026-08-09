import argparse
import json
import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import imageio
import lpips
import torch
import torch.nn.functional as F
from tabulate import tabulate
from torch.nn.functional import mse_loss
from torchvision import transforms
from tqdm import trange

lpips_model = lpips.LPIPS(net="vgg").cuda()


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_video(video_path):
    """
    Load a video and return a PyTorch tensor using imageio (without cv2)

    Parameters:
        video_path (str): Path to the video file

    Returns:
        video_tensor (torch.Tensor): shape -> (num_frames, channels, height, width)
    """
    reader = imageio.get_reader(video_path)
    frames = []
    to_tensor = transforms.ToTensor()

    for frame in reader:
        # imageio gives RGB frames by default
        frame_tensor = to_tensor(frame)  # (C, H, W)
        frames.append(frame_tensor)

    reader.close()

    if not frames:
        raise ValueError(f"Fail to load frames from {video_path}.")

    video_tensor = torch.stack(frames)  # (T, C, H, W)
    return video_tensor


def calculate_ssim(img1, img2):
    """
    Compute SSIM for a single frame using PyTorch.
    """
    C1 = 0.01**2
    C2 = 0.03**2

    # Compute means
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)

    # Compute variances and covariances
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1**2
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1 * mu2

    # SSIM calculation
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def compute_quantization_error(video1_tensor, video2_tensor, frame_interval=1):
    """
    Calculate MSE, PSNR, SSIM, and LPIPS between two videos using PyTorch.

    Parameters:
        video1_tensor (torch.Tensor): shape -> (num_frames, channels, height, width)
        video2_tensor (torch.Tensor): shape -> (num_frames, channels, height, width)
        frame_interval (int): Number of frames per interval. If > 1, returns metrics
                              for each interval. Default is 1 (single average).

    Returns:
        dict: A dictionary containing 'MSE', 'PSNR', 'SSIM', and 'LPIPS' values.
              If frame_interval > 1, values are lists of metrics per interval.
    """
    # Ensure the two videos have the same shape
    assert (
        video1_tensor.shape == video2_tensor.shape
    ), f"Videos must have the same shape. {video1_tensor.shape} != {video2_tensor.shape}"
    num_frames, channels, height, width = video1_tensor.shape

    # Per-frame metrics
    mse_values = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for i in trange(1, num_frames):
        frame1 = video1_tensor[i].unsqueeze(0)  # Add batch dimension
        frame2 = video2_tensor[i].unsqueeze(0)

        # Calculate MSE
        mse = mse_loss(frame1, frame2, reduction="mean")
        mse_values.append(mse.item())

        # Calculate PSNR
        max_pixel_value = 1.0  # Assuming input tensors are normalized to [0, 1]
        psnr = 10 * torch.log10(max_pixel_value**2 / mse)
        psnr_values.append(psnr.item())

        # Calculate SSIM
        ssim = calculate_ssim(frame1, frame2)
        ssim_values.append(ssim.item())

        # Calculate LPIPS
        lpips_value = lpips_model(frame1, frame2)
        lpips_values.append(lpips_value.item())

    # If frame_interval == 1, return single average (original behavior)
    if frame_interval == 1:
        metrics = {
            "MSE": sum(mse_values) / len(mse_values),
            "PSNR": sum(psnr_values) / len(psnr_values),
            "SSIM": sum(ssim_values) / len(ssim_values),
            "LPIPS": sum(lpips_values) / len(lpips_values),
        }
    else:
        # Compute metrics for each interval
        def chunk_average(values, interval):
            """Split values into chunks of size `interval` and compute average for each."""
            averages = []
            for start in range(0, len(values), interval):
                chunk = values[start : start + interval]
                if chunk:
                    averages.append(sum(chunk) / len(chunk))
            return averages

        metrics = {
            "MSE": chunk_average(mse_values, frame_interval),
            "PSNR": chunk_average(psnr_values, frame_interval),
            "SSIM": chunk_average(ssim_values, frame_interval),
            "LPIPS": chunk_average(lpips_values, frame_interval),
        }

    # Return results as a dictionary
    return metrics


def encode_video_with_vae(video_tensor, vae_model):
    """
    Use VAE to encode videos

    paras:
        video_tensor (torch.Tensor)
        vae_model (torch.nn.Module)

    return:
        encoded_video (torch.Tensor)
    """
    vae_model.eval()
    with torch.no_grad():
        # Treat video tensor as batch for encoding
        encoded_video = vae_model.encode(video_tensor)
    return encoded_video


def compute_quantization_error_after_vae(video1_tensor, video2_tensor, vae_model):
    """
    Compute MSE and PSNR with VAE encoding

    paras:
        video1_tensor (torch.Tensor)
        video2_tensor (torch.Tensor)
        vae_model (torch.nn.Module)

    return:
        average_mse (float)
        psnr (float)
    """
    # Encode both videos with VAE
    encoded_video1 = retrieve_latents(
        encode_video_with_vae(video1_tensor, vae_model), sample_mode="argmax"
    )
    encoded_video2 = retrieve_latents(
        encode_video_with_vae(video2_tensor, vae_model), sample_mode="argmax"
    )

    mse = torch.mean((encoded_video1 - encoded_video2) ** 2)
    # Calculate peak signal-to-noise ratio
    psnr = 20 * torch.log10(
        1.0 / torch.sqrt(mse)
    )  # Assume encoded values are in [0, 1]

    return mse.item(), psnr.item()


if __name__ == "__main__":

    # Use argparse to parse arguments
    parser = argparse.ArgumentParser(description="Compute video loss metrics.")
    parser.add_argument(
        "--video1_path",
        "--v1",
        type=str,
        required=True,
        help="Path to the first video.",
    )
    parser.add_argument(
        "--video2_path",
        "--v2",
        type=str,
        required=True,
        help="Path to the second video.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to the VAE model."
    )
    parser.add_argument(
        "--prompt_idx", type=int, default=None, help="Start index of the video."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=0,
        help="Number of frames to skip at the beginning. Default is 0.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Number of frames per interval for computing metrics. "
        "If > 1, outputs metrics for each interval. Default is 1 (single average).",
    )

    args = parser.parse_args()
    video1_path = args.video1_path
    video2_path = args.video2_path
    # Load videos
    video1_tensor = load_video(video1_path)
    video2_tensor = load_video(video2_path)

    video1_tensor, video2_tensor = video1_tensor.cuda(), video2_tensor.cuda()

    # Skip the first N frames if specified
    if args.skip_frames > 0:
        video1_tensor = video1_tensor[args.skip_frames :]
        video2_tensor = video2_tensor[args.skip_frames :]
        print(f"Skipped first {args.skip_frames} frames.")

    print(f"video tensor shape: {video1_tensor.shape}")

    # Calculate direct comparison error
    metrics = compute_quantization_error(
        video1_tensor, video2_tensor, frame_interval=args.frame_interval
    )

    if args.frame_interval == 1:
        # Single average output as table
        table_data = [
            ["MSE", f"{metrics['MSE']:.6f}"],
            ["PSNR", f"{metrics['PSNR']:.2f} dB"],
            ["SSIM", f"{metrics['SSIM']:.4f}"],
            ["LPIPS", f"{metrics['LPIPS']:.4f}"],
        ]
        print("\nAverage Video Metrics:")
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
    else:
        # Per-interval output as table
        num_intervals = len(metrics["PSNR"])
        print(
            f"\nComputed metrics for {num_intervals} intervals (every {args.frame_interval} frames):"
        )
        table_data = []
        for i in range(num_intervals):
            frame_start = i * args.frame_interval + 1
            frame_end = min((i + 1) * args.frame_interval, video1_tensor.shape[0] - 1)
            table_data.append(
                [
                    f"{i}",
                    f"{frame_start}-{frame_end}",
                    f"{metrics['MSE'][i]:.6f}",
                    f"{metrics['PSNR'][i]:.2f}",
                    f"{metrics['SSIM'][i]:.4f}",
                    f"{metrics['LPIPS'][i]:.4f}",
                ]
            )
        headers = ["Interval", "Frames", "MSE", "PSNR (dB)", "SSIM", "LPIPS"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Update idx and seed
    if args.prompt_idx is not None:
        metrics["idx"] = args.prompt_idx
    if args.seed is not None:
        metrics["seed"] = args.seed
    if args.frame_interval > 1:
        metrics["frame_interval"] = args.frame_interval

    # Output to the jsonl file.
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        assert args.output_path.endswith(".jsonl"), "Output path must end with .jsonl"
        with open(args.output_path, "a") as f:
            json.dump(metrics, f)
            f.write("\n")
