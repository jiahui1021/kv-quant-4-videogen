"""CPU tests for the Forcing-KV resident compression calculation."""

import torch

from tools.forcing_kv_compression_ratio import measure_cache


def test_ratio_counts_actual_kv_storage_and_full_head_equivalent():
    cache = [
        {
            "sink_k": torch.zeros(1, 2, 3, 4, dtype=torch.bfloat16),
            "sink_v": torch.zeros(1, 2, 3, 4, dtype=torch.bfloat16),
            "local_end_index": torch.tensor(2),
        }
    ]
    report = measure_cache(cache)
    assert report["resident_kv_bytes"] == 2 * 2 * 3 * 4 * 2
    assert report["bf16_equivalent_bytes"] == 1 * 2 * 3 * 4 * 2 * 2
    assert report["compression_ratio"] == 1.0


def test_empty_cache_is_explicitly_unavailable():
    report = measure_cache([])
    assert report["compression_ratio"] is None
