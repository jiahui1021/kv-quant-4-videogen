# Upstream

- Repository: <https://github.com/guandeh17/Self-Forcing>
- Commit: `33593df3e81fa3ec10239271dd2c100facac6de1` ("Clean", 2025-09-11)
- License: see `LICENSE` (unchanged)

The tree is a verbatim export of that commit, vendored so that the code which
runs is the code in this repository (previously it was cloned and patched on
each machine, which let server copies drift from the repository).

## Local modifications

`pipeline/causal_inference.py`: the block loop shows a `tqdm` progress bar
(one step per generated block) instead of printing every denoising timestep.

`wan/modules/causal_model.py` adds the KV-cache quantization hook to
`CausalWanSelfAttention`:

- `_attention_with_incremental_cache`: append-only cache for quantizers that
  declare `supports_incremental_cache` (RTN, KIVI, QuaRot).  Keys are cached
  after RoPE; history is never re-quantized; only the attention window
  (`start_token`) is decoded.  QuaRot additionally rotates Q/K/V and restores
  the attention output.
- The inline cache path: for the remaining quantizers (the composite methods),
  the active prefix is dequantized, the current block written, and the prefix
  quantized again on every call.

To see the exact change:

```bash
git diff --no-index <upstream-checkout>/wan/modules/causal_model.py wan/modules/causal_model.py
```
