from typing import List, Optional, Tuple
from termcolor import cprint

import torch
import torch.nn as nn
import torch.amp as amp

import numpy as np
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from safetensors.torch import load_file

from .lora_utils import create_lora_network
from ..context_parallel import context_parallel_util
from .attention import Attention, MultiHeadCrossAttention
from .blocks import (
    TimestepEmbedder,
    CaptionEmbedder,
    PatchEmbed3D,
    FeedForwardSwiGLU,
    FinalLayer_FP32,
    LayerNorm_FP32,
    modulate_fp32,
)

from quant_videogen.timer import time_logging_decorator
from quant_videogen.compress import get_quantize_fn, compress_kv_cache
from .utils import _onload_kv_cache, _offload_kv_cache


class LongCatSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw=None,
        layer_idx=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        # scale and gate modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )

        self.mod_norm_attn = LayerNorm_FP32(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.mod_norm_ffn = LayerNorm_FP32(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.pre_crs_attn_norm = LayerNorm_FP32(
            hidden_size, eps=1e-6, elementwise_affine=True
        )

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw,
            layer_idx=layer_idx,
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.ffn = FeedForwardSwiGLU(
            dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio)
        )

    def forward(
        self,
        x,
        y,
        t,
        y_seqlen,
        latent_shape,
        num_cond_latents=None,
        return_kv=False,
        kv_cache=None,
        skip_crs_attn=False,
        timestep_int=None,
    ):
        """
        x: [B, N, C]
        y: [1, N_valid_tokens, C]
        t: [B, T, C_t]
        y_seqlen: [B]; type of a list
        latent_shape: latent shape of a single item
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape  # S != T*H*W in case of CP split on H*W.

        with time_logging_decorator("Shift Scale Gate", logging_level=2):
            # compute modulation params in fp32
            with amp.autocast(device_type="cuda", dtype=torch.float32):
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
                )  # [B, T, 1, C]

        with time_logging_decorator("Modulated Self Attention", logging_level=2):
            # self attn with modulation
            x_m = modulate_fp32(
                self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa
            ).view(B, N, C)

        with time_logging_decorator("Self Attention", logging_level=2):
            if kv_cache is not None:
                # kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
                kv_cache = _onload_kv_cache(kv_cache, x.device)
                attn_outputs = self.attn.forward_with_kv_cache(
                    x_m,
                    shape=latent_shape,
                    num_cond_latents=num_cond_latents,
                    kv_cache=kv_cache,
                    timestep_int=timestep_int,
                )
            else:
                attn_outputs = self.attn(
                    x_m,
                    shape=latent_shape,
                    num_cond_latents=num_cond_latents,
                    return_kv=return_kv,
                    timestep_int=timestep_int,
                )

            if return_kv:
                x_s, kv_cache = attn_outputs
            else:
                x_s = attn_outputs

        with time_logging_decorator("Gate Self Attention", logging_level=2):
            with amp.autocast(device_type="cuda", dtype=torch.float32):
                x = x + (gate_msa * x_s.view(B, -1, N // T, C)).view(
                    B, -1, C
                )  # [B, N, C]
            x = x.to(x_dtype)

        with time_logging_decorator("Cross Attention", logging_level=2):
            # cross attn
            if not skip_crs_attn:
                if kv_cache is not None:
                    num_cond_latents = None
                x = x + self.cross_attn(
                    self.pre_crs_attn_norm(x),
                    y,
                    y_seqlen,
                    num_cond_latents=num_cond_latents,
                    shape=latent_shape,
                )

        with time_logging_decorator("Modulated Feed Forward", logging_level=2):
            # ffn with modulation
            x_m = modulate_fp32(
                self.mod_norm_ffn, x.view(B, -1, N // T, C), shift_mlp, scale_mlp
            ).view(B, -1, C)

        with time_logging_decorator("Feed Forward", logging_level=2):
            x_s = self.ffn(x_m)

        with time_logging_decorator("Gate Feed Forward", logging_level=2):
            with amp.autocast(device_type="cuda", dtype=torch.float32):
                x = x + (gate_mlp * x_s.view(B, -1, N // T, C)).view(
                    B, -1, C
                )  # [B, N, C]
            x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        else:
            return x


class LongCatVideoTransformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        # default params
        patch_size: Tuple[int] = (1, 2, 2),
        # attention config
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None,
        text_tokens_zero_pad: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(
            t_embed_dim=adaln_tembed_dim,
            frequency_embedding_size=frequency_embedding_size,
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw,
                    layer_idx=i,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer_FP32(
            hidden_size,
            np.prod(self.patch_size),
            out_channels,
            adaln_tembed_dim,
        )

        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

        self.lora_dict = {}
        self.active_loras = []

        self.quant_config = None

    def load_lora(
        self,
        lora_path,
        lora_key,
        multiplier=1.0,
        lora_network_dim=128,
        lora_network_alpha=64,
    ):
        lora_network_state_dict_loaded = load_file(lora_path, device="cpu")
        lora_network = create_lora_network(
            transformer=self,
            lora_network_state_dict_loaded=lora_network_state_dict_loaded,
            multiplier=multiplier,
            network_dim=lora_network_dim,
            network_alpha=lora_network_alpha,
        )

        lora_network.load_state_dict(lora_network_state_dict_loaded, strict=True)

        self.lora_dict[lora_key] = lora_network

    def enable_loras(self, lora_key_list=[]):
        self.disable_all_loras()

        module_loras = {}  # {module_name: [lora1, lora2, ...]}
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        for lora_key in lora_key_list:
            if lora_key in self.lora_dict:
                for lora in self.lora_dict[lora_key].loras:
                    lora.to(model_device, dtype=model_dtype, non_blocking=True)
                    module_name = lora.lora_name.replace(
                        "lora___lorahyphen___", ""
                    ).replace("___lorahyphen___", ".")
                    if module_name not in module_loras:
                        module_loras[module_name] = []
                    module_loras[module_name].append(lora)
                self.active_loras.append(lora_key)

        for module_name, loras in module_loras.items():
            module = self._get_module_by_name(module_name)
            if not hasattr(module, "org_forward"):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)

    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)

            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = (
                        lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
                    )
                    total_lora_output += lora_output

            return org_output + total_lora_output

        return multi_lora_forward

    def _get_module_by_name(self, module_name):
        try:
            module = self
            for part in module_name.split("."):
                module = getattr(module, part)
            return module
        except AttributeError as e:
            raise ValueError(f"Cannot find module: {module_name}, error: {e}")

    def disable_all_loras(self):
        for name, module in self.named_modules():
            if hasattr(module, "org_forward"):
                module.forward = module.org_forward
                delattr(module, "org_forward")

        for lora_key, lora_network in self.lora_dict.items():
            for lora in lora_network.loras:
                lora.to("cpu")

        self.active_loras.clear()

    def enable_bsa(
        self,
    ):
        for block in self.blocks:
            block.attn.enable_bsa = True

    def disable_bsa(
        self,
    ):
        for block in self.blocks:
            block.attn.enable_bsa = False

    @time_logging_decorator("Model Forward", logging_level=0)
    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        return_kv=False,
        kv_cache_dict={},
        skip_crs_attn=False,
        offload_kv_cache=False,
        timestep_int=None,
    ):

        B, _, T, H, W = hidden_states.shape

        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        assert (
            self.patch_size[0] == 1
        ), "Currently, 3D x_embedder should not compress the temporal dimension."

        # expand the shape of timestep from [B] to [B, T]
        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)  # [B, T]

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)  # [B, N, C]

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            t = self.t_embedder(
                timestep.float().flatten(), dtype=torch.float32
            ).reshape(
                B, N_t, -1
            )  # [B, T, C_t]

        encoder_hidden_states = self.y_embedder(
            encoder_hidden_states
        )  # [B, 1, N_token, C]

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = (
                encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            )
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(
                encoder_attention_mask.dtype
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )  # [1, N_valid_tokens, C]
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()  # [B]
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[
                0
            ]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(
                1, -1, hidden_states.shape[-1]
            )

        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            hidden_states = rearrange(
                hidden_states, "B (T H W) C -> B T H W C", T=N_t, H=N_h, W=N_w
            )
            hidden_states = context_parallel_util.split_cp_2d(
                hidden_states, seq_dim_hw=(2, 3), split_hw=self.cp_split_hw
            )
            hidden_states = rearrange(hidden_states, "B T H W C -> B (T H W) C")

        # blocks
        kv_cache_dict_ret = {}
        for i, block in enumerate(self.blocks):
            # # print the memory usage in GB
            # print(
            #     f"Memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB at layer {i}"
            # )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_outputs = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    (N_t, N_h, N_w),
                    num_cond_latents,
                    return_kv,
                    kv_cache_dict.get(i, None),
                    skip_crs_attn,
                )
            else:
                with time_logging_decorator("Block Forward", logging_level=1):
                    block_outputs = block(
                        hidden_states,
                        encoder_hidden_states,
                        t,
                        y_seqlens,
                        (N_t, N_h, N_w),
                        num_cond_latents,
                        return_kv,
                        kv_cache_dict.get(i, None),
                        skip_crs_attn,
                        timestep_int=timestep_int,
                    )

            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (
                        kv_cache[0].contiguous(),
                        kv_cache[1].contiguous(),
                    )
            else:
                hidden_states = block_outputs

        hidden_states = self.final_layer(
            hidden_states, t, (N_t, N_h, N_w)
        )  # [B, N, C=T_p*H_p*W_p*C_out]

        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            hidden_states = context_parallel_util.gather_cp_2d(
                hidden_states, shape=(N_t, N_h, N_w), split_hw=self.cp_split_hw
            )

        hidden_states = self.unpatchify(
            hidden_states, N_t, N_h, N_w
        )  # [B, C_out, H, W]

        # cast to float32 for better accuracy
        hidden_states = hidden_states.to(torch.float32)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        else:
            return hidden_states

    def unpatchify(self, x, N_t, N_h, N_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

    def quantize_kv_cache(self, kv_cache_dict, offload_kv_cache=False):

        # Record the time using torch.cuda.Event
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # Do nothing if quantization type is none
        if self.quant_config.quant_type == "none":
            self._print_memory_usage(kv_cache_dict)
            cprint("No quantization is applied. Returning original KV cache.", "light_blue")

            return kv_cache_dict

        cprint(f"Quantizing kv cache with config:\n{self.quant_config}", "light_blue")

        with torch.no_grad():
            for layer_idx, (k, v) in kv_cache_dict.items():
                # ==========================================================
                # Define the quantization function (easily swappable)
                # ==========================================================
                # To use a different quantization method, simply replace this function.
                # The function should take a tensor and return the quantized tensor.

                quantize_fn = get_quantize_fn(self.quant_config.quant_type, self.quant_config)

                k, v = k.contiguous(), v.contiguous()

                if offload_kv_cache:
                    # k, v = k.to("cuda"), v.to("cuda")  # TODO: Support Multiple GPUs
                    k, v = _onload_kv_cache((k, v), "cuda")

                k_quant, v_quant = compress_kv_cache(
                    k, v, self.quant_config.quant_type, self.quant_config, quantize_fn
                )

                k_quant, v_quant = self._pack_info_into_kv_cache(k_quant, v_quant, k.dtype)

                self._print_kv_cache_mse_error(k, k_quant, v, v_quant, layer_idx)

                if offload_kv_cache:
                    # k_quant, v_quant = k_quant.to("cpu"), v_quant.to("cpu")
                    k_quant, v_quant = _offload_kv_cache((k_quant, v_quant))

                kv_cache_dict[layer_idx] = (k_quant, v_quant)

        # Record the end time using torch.cuda.Event
        end_time.record()
        torch.cuda.synchronize()
        duration = start_time.elapsed_time(end_time)
        cprint(f"Quantization KV Cache Time: {(duration / 1000):.2f} s", "light_cyan")

        self._print_memory_usage(kv_cache_dict)

        return kv_cache_dict


    def _print_memory_usage(self, kv_cache_dict):
        """ Onload all KV cache to check its memory usage """
        
        torch.cuda.empty_cache()

        MemoryUsage = []
        
        memory_usage_before_onload = torch.cuda.memory_allocated() / 1024 ** 2
        MemoryUsage.append(memory_usage_before_onload)
        for layer_idx, (k, v) in kv_cache_dict.items():
            _memory = torch.cuda.memory_allocated() / 1024 ** 2

            kv_cache_dict[layer_idx] = _onload_kv_cache(kv_cache_dict[layer_idx], "cuda")

            memory_usage_after_onload = torch.cuda.memory_allocated() / 1024 ** 2
            MemoryUsage.append(memory_usage_after_onload)
            # print(f"Per Layer {layer_idx:02d} Memory Usage: {(memory_usage_after_onload - _memory):.2f} MB | Before Onload: {_memory:.2f} MB | After Onload: {memory_usage_after_onload:.2f} MB")
            
        
        # Print Statistics of Memory Usage. Get peak memory usage and per layer memory usage. Total memory usage is per layer * num_layers
        peak_memory_usage = max(MemoryUsage)
        diff_per_layer = [MemoryUsage[i+1] - MemoryUsage[i] for i in range(len(MemoryUsage)-1)]
        per_layer_memory_usage = np.median(diff_per_layer)
        total_kv_cache_memory_usage = per_layer_memory_usage * len(kv_cache_dict)

        cprint(f"Peak Memory Usage: {peak_memory_usage:.2f} MB | Other Memory Usage: {MemoryUsage[0]:.2f} MB | Per Layer Memory Usage: {per_layer_memory_usage:.2f} MB | Total KV Cache Memory Usage: {total_kv_cache_memory_usage:.2f} MB", "light_blue")

        for layer_idx, (k, v) in kv_cache_dict.items():
            kv_cache_dict[layer_idx] = _offload_kv_cache(kv_cache_dict[layer_idx])
            
            memory_usage_after_offload = torch.cuda.memory_allocated() / 1024 ** 2
            # print(f"Memory usage before offload: {memory_usage_after_onload:.2f} MB | Memory usage after offload: {memory_usage_after_offload:.2f} MB")
            
        torch.cuda.empty_cache()
        
    
    def _print_kv_cache_mse_error(self, k, k_quant, v, v_quant, layer_idx):
        """ Print the Rel L2 error between the original and quantized KV cache """

        if isinstance(k_quant, dict) and isinstance(v_quant, dict):
            return
        
        k_rel_l2 = torch.norm(k - k_quant, p=2) / torch.norm(k, p=2)
        v_rel_l2 = torch.norm(v - v_quant, p=2) / torch.norm(v, p=2)
        cprint(f"Layer {layer_idx:02d} | K Rel L2: {k_rel_l2:.4f} | V Rel L2: {v_rel_l2:.4f}", "light_blue")

    
    def _pack_info_into_kv_cache(self, k_cache, v_cache, output_dtype):
        """ Pack with all information with KV cache. Only happens when the tensor is real quantized. """
        if isinstance(k_cache, dict) and isinstance(v_cache, dict):
            k_cache["info"] = {
                "output_dtype": output_dtype,
                "quant_config": self.quant_config,
            }
            v_cache["info"] = {
                "output_dtype": output_dtype,
                "quant_config": self.quant_config,
            }

        return k_cache, v_cache