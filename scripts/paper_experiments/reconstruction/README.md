# Cross-method KV reconstruction error

> **One collected trace, every method, one measurement path.**

BF16, RTN, KIVI, QuaRot, Hadamard-K, QVG and TempoKV are all scored on the
*same* clean K/V tensors, reporting the error at each frame index as well as
the pooled figure. A difference between the numbers is therefore a difference
between the methods and not between two measurement harnesses.

The comparison lives in this repository because five of the six method families
are implemented here (`kv_quant/` and `third_party/Quant-VideoGen/`). TempoKV is
imported from its own checkout via `--tempokv-root`.

## Step 1 — collect one trace

Trace collection lives in the TempoKV repository, which already owns the
Self-Forcing trace bridge. For the 1401-frame horizon:

```bash
cd ../Tempokv
CHECKPOINT=/models/self_forcing_dmd.pt \
WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \
PROMPTS=assets/moviegen128_first30.txt \
bash scripts/paper_experiments/long_horizon/collect_trace.sh
```

That writes `data/long-horizon/1401f/clean_trace.npz` (1401 video frames = 351
latent frames; see that script's README for the conversion). Any trace in the
`[layers, records, frames, heads, dim]` layout works here, including the
standard short calibration trace.

## Step 2 — score every method

```bash
python scripts/paper_experiments/reconstruction/run_reconstruction.py \
    --trace             ../Tempokv/data/long-horizon/1401f/clean_trace.npz \
    --calibration-trace ../Tempokv/data/self-forcing/trace/clean_trace.npz \
    --tempokv-root      ../Tempokv \
    --output-dir        results/reconstruction/1401f \
    --methods bf16 rtn_int2 kivi_int2 quarot_int2 hadamard_int2 qvg_int2 tempokv_int2
```

| method | source | needs |
| --- | --- | --- |
| `bf16` | this module | nothing; the bfloat16 storage floor |
| `rtn_int2` / `rtn_int4` | `kv_quant.rtn` | nothing |
| `kivi_int2` / `kivi_int4` | `kv_quant.kivi` | nothing |
| `quarot_int2` / `quarot_int4` | `kv_quant.quarot_kv` | nothing |
| `hadamard_int2` / `hadamard_int4` | `kv_quant.quarot_kv` | nothing |
| `qvg_int2` / `qvg_int4` | `third_party/Quant-VideoGen` | **Triton** (CUDA) |
| `tempokv_int2` / `tempokv_int4` | the TempoKV checkout | `--tempokv-root` **and** `--calibration-trace` |

### Two asymmetries worth stating in the paper

- **TempoKV fits predictors; the others do not.** It is therefore given a
  *separate* calibration trace and scored on the evaluation trace. Fitting and
  scoring on the same trace would flatter it against methods that never see the
  data twice, so `--calibration-trace` is required rather than optional for it.
- **BF16 is not a no-op.** Every other method also stores its reconstruction in
  bfloat16, so BF16 is the error floor none of them can beat, not a zero line.

### Methods that cannot run here

A method that cannot run reports `status: unavailable` with the reason, and the
figure names it under the title. It is never silently dropped — an absent curve
would read as an untested method. QVG's published codec is a Triton kernel, so
a CPU-only host genuinely cannot run it; the simulation path is **not**
substituted and labelled "QVG".

## Step 3 — figure

```bash
python scripts/paper_experiments/reconstruction/plot_figure.py \
    --report     results/reconstruction/1401f/reconstruction.json \
    --output-dir results/reconstruction/1401f/figure
```

Four panels: K and V, per-frame and cumulative. The cumulative curve's last
point **is** the pooled number, so a table and the plot cannot disagree.

## Reading the results

Per method and tensor the report carries `per_frame`, `cumulative`, and a
`summary` with `first_frame_nmse`, `last_frame_nmse`, `max_frame_nmse`,
`head_mean_nmse`, `tail_mean_nmse` and `tail_over_head`.

`tail_over_head` is the long-horizon question in one number: mean error over
the last `--tail-fraction` of the sequence divided by the mean over the rest.
Near 1.0 means the method holds up as the horizon extends; well above 1.0 means
it degrades.

## Tensor layout

The trace is `[layers, records, frames, heads, dim]`. One layer of it is
`[B, L, H, D]`, which is exactly what every `kv_quant` quantizer expects, so no
permute is needed there — that shared convention is why one trace can feed all
of them. QVG is driven in `[B, H, L, D]` and gets the same permute the
Causal-Forcing runtime applies. TempoKV takes the five-dimensional trace as it
stands.

`--max-records N` trims the record axis to bound memory; records are
independent rollouts, so this scales memory down without shortening the frame
axis the curve is measured along.

## Minimal complete run

```bash
python scripts/paper_experiments/reconstruction/run_minimal.py \
    --output-root /tmp/reconstruction-minimal
```

CPU, seconds, synthetic traces. It checks that every requested method either
runs or declines with a reason — a method reported as `failed` fails the run.
Its numbers describe the synthetic generator and say nothing about any method.
