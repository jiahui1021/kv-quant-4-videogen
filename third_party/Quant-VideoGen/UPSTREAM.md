# Quant-VideoGen upstream

- upstream repository: https://github.com/svg-project/Quant-VideoGen
- upstream commit: `0601468f2dbba6a17ac7086faec6d41527cad188`
- purpose: official QVG baseline for Causal-Forcing
- copied on: 2026-08-14
- copied files: `quant_videogen/` (26 source files)
- local modifications: `ChunkedKVCache` uses a single-state decompressor for
  K/V reads and rejects writes into immutable quantized chunks; the adapter
  records the rationale in its debug tests.

The directory is copied from the commit above, with the two small cache
lifecycle fixes documented above. Causal-Forcing-specific integration lives
outside this directory.
