# Self-Forcing integration

The KV-cache quantizers live in [`../kv_quant/`](../kv_quant/); this directory
holds the Self-Forcing runner, the evaluation scripts and the vendored
upstream model under `third_party/Self-Forcing`.

## Pipeline

```text
scripts/01_generate.py        generate videos with BF16 or a quantized cache
scripts/02_eval_fidelity.py   PSNR / SSIM / LPIPS against the BF16 run
scripts/03_eval_vbench.sh     VBench
scripts/04_eval_drift_curve.py  imaging quality over time
scripts/05_summarize_results.py tables from the metrics files
```

Launchers: `06_run_baseline_matrix.sh` (BF16 plus the baselines),
`07_run_paper_baselines.sh` (the six paper rows against an existing BF16 run),
`07_run_generation_multi_gpu.sh` (one method per GPU),
`08_run_evaluation_suite.sh`, `09_run_full_research_pipeline.sh`.

StoryEval has its own three scripts: `run_storyeval.py`,
`eval_storyeval_vbench.py` / `eval_storyeval_drift.py`, `summarize_storyeval.py`.

## One run

```bash
python scripts/01_generate.py \
  --method KIVI_INT2 \
  --checkpoint-path checkpoints/self_forcing_dmd.pt \
  --prompt-path prompts/moviegen_128.txt \
  --num-output-frames 180 \
  --results-root results
```

Defaults match the Tempokv and QVG runners: MovieGen-128 prompts, noise seed
`seed + prompt_index * 1000003`, full-history attention, 180 latent frames.
Videos land in `results/videos/<METHOD>/`, metrics in `results/metrics/`.

## Efficiency and latency

`scripts/paper_experiments/` holds the efficiency runner and the latency
harness; `scripts/probe_horizon.py` finds the longest horizon that fits in
memory for one method.
