# Self-Forcing RTN latency adapter

`run_rtn.py` provides the `RTN_INT2` and `RTN_INT4` entries for TempoKV's
paper latency coordinator. Its formal entry is
`Self-Forcing/scripts/01_generate.py`; it reuses that pipeline and the
repository's `RTNQuantizer`.

## Protocol

- `--phase pair` loads the model once, runs one complete warmup, discards it,
  then measures one complete generation. It writes canonical reports to
  `PAPER_EXPERIMENT_WARMUP_REPORT` and
  `PAPER_EXPERIMENT_MEASURE_REPORT`. Both reports return the same coordinator-
  assigned `PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID` from that one loaded process.
- `--phase compilation` executes a cold and hot generation after model load.
  `one_time_compilation_s` is the non-negative cold-minus-hot difference, so
  model loading is excluded and first-use kernel compilation is reported
  separately from E2E.
- E2E starts immediately before `pipeline.inference()` and ends after the
  final RTN cache commit. It contains VAE decode, but excludes model loading,
  compilation accounting, CPU transfer, and MP4 writing.
- Formal mode synchronizes CUDA before and after E2E, VAE decode, every RTN
  cache encode, and every RTN cache decode.
- The adapter observes the prompt hash, GPU name/UUID, CUDA version and
  attention backend from the actual runtime, then compares them with
  `PAPER_EXPERIMENT_FROZEN_JSON`.
- The prompt hash is SHA-256 of the selected prompt text in UTF-8, without the
  file's trailing newline. The attention backend is reported as
  `torch_compiled_flex_attention`.
- Formal runs override `local_attn_size` to the requested latent horizon and
  verify it after pipeline construction. The shared lifecycle label is
  `self_forcing_global_causal_attention_full_resident_history_read_each_denoising_step`.

Copy the two command arrays from `manifest_methods.example.json` into the
corresponding entries of TempokV's latency manifest. The paired coordinator
must call the command with `phase=pair`; separate warmup and measure process
invocations are intentionally unsupported.

The prompt file must contain exactly one prompt. The current Self-Forcing VAE
maps latent length `L` to `4L-3` pixel frames, and the selected chunk-wise
config also requires `L` to follow the model's 3-frame generation block. Unsupported exact
paper horizons are rejected instead of being rounded or mislabeled.
