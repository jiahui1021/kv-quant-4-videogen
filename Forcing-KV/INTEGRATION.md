# Forcing-KV integration

This directory is a source-only snapshot of the local Forcing-KV repository.
The source snapshot came from local commit `50ac1f4` after the original
Forcing-KV remote was removed. Its Git history remains in
`/Users/hui/Desktop/code/Forcing-KV`; this copy is part of KVQuant4Video and
does not contain a nested `.git` directory.

Generated videos, checkpoints, evaluation results, and other binary artifacts
were intentionally left out of this copy. The runnable SF/CF smoke entry point
is:

```bash
Forcing-KV/scripts/run_sf_cf_717.sh --only both
```

It runs one 180-latent-frame / 717-pixel-frame video for Self-Forcing and
Causal-Forcing, then writes one report per workload under
`efficiency/compression_ratio_0.json`.

The reported ratio is:

```text
BF16-equivalent resident bytes / actual resident KV tensor bytes
```

The denominator is obtained by walking the live K/V tensors, including the
head-group and dynamic-history buffers. The numerator represents the same
resident logical token positions with all heads stored as BF16.
