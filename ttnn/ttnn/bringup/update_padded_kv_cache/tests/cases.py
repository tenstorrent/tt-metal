# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.update_padded_kv_cache call a model makes (bringup-fork-tests skill). Append
only; never edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json). Each
case is literal data; a family's calls differ only in their chunk start / valid end, listed per signature below."""

# DeepSeek-V4.1-Flash on 2x2 (O.1, capture 2026-10-27): the served KV caches (tt/served.py alloc_served): bf16
# ROW_MAJOR [1, 1, rows, width] a chip in DRAM, ND-sharded one 32-token record [1, 1, 32 / ratio, width] per shard,
# round-robin over every DRAM bank; TP-dedup caches (tp_axis 1) carry a Shard(2) x Shard(2) topology, the others a
# replicated one. The input: [1, 1, W, width] bf16 ROW_MAJOR DRAM interleaved a chip, mesh row r's rows (the server's
# order), replicated over the columns. slot 0, layer 0 of 1 (the ladder's own one-slab caches).
_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "O.1",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576, "fabric_payload_bytes": 10880},
    "cluster_axis": 0,
    "slot_idx": 0,
    "layer_idx": 0,
    "num_layers": 1,
    "exact": True,
}


def _rm(shape, memory_layout, **kw):
    return {
        "shape": shape,
        "dtype": "BFLOAT16",
        "layout": "ROW_MAJOR",
        "buffer": "DRAM",
        "memory_layout": memory_layout,
        **kw,
    }


# 4_index_k_r1 (tt/compressor.py TtServedRows ratio 1, dedup=False): 27 chunks of 1024 rows a chip, scalar path
_IDX_R1 = {
    **_DS41,
    "cache": _rm([1, 1, 27648, 128], "ND_SHARDED", shard_rows=32, dedup=False),
    "input": _rm([1, 1, 1024, 128], "INTERLEAVED"),
    "tp_axis": None,
    "path": "scalar",
}
# 0_window_kv (tt/attention.py TtWindowCache) and 3_compress_kv_r1 (TtServedRows ratio 1, dedup=True; the same call):
# TP-dedup, stripes of 512 rows, 27 chunks, scalar path
_WIN = {
    **_DS41,
    "cache": _rm([1, 1, 13824, 512], "ND_SHARDED", shard_rows=32, dedup=True),
    "input": _rm([1, 1, 1024, 512], "INTERLEAVED"),
    "tp_axis": 1,
    "path": "scalar",
}
# Ratio-2 caches (TtServedRows ratio 2): the metadata path, slot / start / valid as 1-element uint32 [1, 1] ROW_MAJOR
# DRAM tensors, in compressed rows. Their values are not in the capture (device tensors): each case takes a start that
# is a multiple of 16 rows but not of a stripe (the model's ratio-2 starts: start token / 2), and a valid end (ceil32 of
# the end token / 2, a multiple of 16) that ends inside a 32-row block, so the op's 32-row clamp rounding is exercised.
_META = _rm([1, 1], "INTERLEAVED", dtype="UINT32")
# 1_compress_kv_r2 (dedup=True): stripes of 256 compressed rows, 6912 rows a chip
_CKV_R2 = {
    **_DS41,
    "cache": _rm([1, 1, 6912, 512], "ND_SHARDED", shard_rows=16, dedup=True),
    "input": _rm([1, 1, 512, 512], "INTERLEAVED"),
    "meta": _META,
    "tp_axis": 1,
    "path": "metadata",
}
# 2_index_k_r2 (dedup=False): 512 compressed rows a chip a chunk, 13824 rows a chip
_IDX_R2 = {
    **_DS41,
    "cache": _rm([1, 1, 13824, 128], "ND_SHARDED", shard_rows=16, dedup=False),
    "input": _rm([1, 1, 512, 128], "INTERLEAVED"),
    "meta": _META,
    "tp_axis": None,
    "path": "metadata",
}

# (sig, kv_actual_global, valid_global) as captured
_IDX_R1_CALLS = [
    ("52ea6200be", 0, 2048),
    ("c4a243616f", 2048, 4096),
    ("71130923ec", 4096, 6144),
    ("b7d1382051", 6144, 8192),
    ("f994267b58", 8192, 10240),
    ("4ffbfe5765", 10240, 12288),
    ("8901eca48a", 12288, 14336),
    ("09a0901c6f", 14336, 16384),
    ("b85c0a4b63", 16384, 18432),
    ("d427c81ecd", 18432, 20480),
    ("21d5269989", 20480, 22528),
    ("3fb5b44e12", 22528, 24576),
    ("a80dc628a8", 24576, 26624),
    ("434760527d", 26624, 28672),
    ("13b4ed2c72", 28672, 30720),
    ("bcc83e2268", 30720, 32768),
    ("f165bdb517", 32768, 34816),
    ("65944b5dba", 34816, 36864),
    ("9b5c178c23", 36864, 38912),
    ("f39007ffe1", 38912, 40960),
    ("9a8ffaa5de", 40960, 43008),
    ("89c856865f", 43008, 45056),
    ("0bd5f49e12", 45056, 47104),
    ("7692784720", 47104, 49152),
    ("f4d5ed161f", 49152, 51200),
    ("d204137fd4", 51200, 53248),
    ("61f95ca744", 53248, 55296),
]
_WIN_CALLS = [
    ("68249e5812", 0, 2048),
    ("c9538c2977", 2048, 4096),
    ("2f30af7fbb", 4096, 6144),
    ("47e6b437ae", 6144, 8192),
    ("326396972c", 8192, 10240),
    ("c8655090e4", 10240, 12288),
    ("f77910dfa7", 12288, 14336),
    ("6df26f6d01", 14336, 16384),
    ("98225c66e9", 16384, 18432),
    ("e8c465d6d2", 18432, 20480),
    ("bb660aaece", 20480, 22528),
    ("70d94d4399", 22528, 24576),
    ("fd9c770cd0", 24576, 26624),
    ("49ea6fba63", 26624, 28672),
    ("98ed43fb15", 28672, 30720),
    ("323f432ac6", 30720, 32768),
    ("b6824d5fc1", 32768, 34816),
    ("519e40bc05", 34816, 36864),
    ("f3f349bccf", 36864, 38912),
    ("38c48fda5d", 38912, 40960),
    ("b8c000ffb3", 40960, 43008),
    ("2e7de1563d", 43008, 45056),
    ("f0dfc5d5da", 45056, 47104),
    ("3fb58fa69b", 47104, 49152),
    ("154e56fe48", 49152, 51200),
    ("6e7149ad1d", 51200, 53248),
    ("ae8e6d3b6d", 53248, 55296),
    # a start inside a stripe (53120 = 25 chunks + 1920): the start's mesh row rotates, its chips straddle a slab
    ("ed25109c8e", 53120, 53248),
]

CASES = (
    [
        {
            **_IDX_R1,
            "id": f"deepseek_v41_flash_d_p-2x2-index-k-r1-w128-scalar-kv{kv}-v{valid}",
            "sig": sig,
            "kv_actual_global": kv,
            "valid_global": valid,
            "seed": 1000 + i,
        }
        for i, (sig, kv, valid) in enumerate(_IDX_R1_CALLS)
    ]
    + [
        {
            **_WIN,
            "id": f"deepseek_v41_flash_d_p-2x2-window-kv-tpdedup-w512-scalar-kv{kv}-v{valid}",
            "sig": sig,
            "kv_actual_global": kv,
            "valid_global": valid,
            "seed": 2000 + i,
        }
        for i, (sig, kv, valid) in enumerate(_WIN_CALLS)
    ]
    + [
        {
            **_CKV_R2,
            "id": "deepseek_v41_flash_d_p-2x2-compress-kv-r2-tpdedup-w512-meta-kv10608-v11312",
            "sig": "0bb231744d",
            # chunk 10 + 368 rows: mesh row 0 (offset 368 in its 512-row window), chip 1 (offset 112 in its stripe);
            # valid 11312 -> rows up to 11328 written (the 32-row clamp)
            "kv_actual_global": 10608,
            "valid_global": 11312,
            "seed": 3000,
        },
        {
            **_IDX_R2,
            "id": "deepseek_v41_flash_d_p-2x2-index-k-r2-w128-meta-kv7888-v8880",
            "sig": "3060c5e5c0",
            # chunk 7 + 720 rows: mesh row 1 (offset 208 in its 512-row window); valid 8880 -> rows up to 8896 written,
            # mesh row 1 spills into the next slab
            "kv_actual_global": 7888,
            "valid_global": 8880,
            "seed": 3001,
        },
    ]
)
