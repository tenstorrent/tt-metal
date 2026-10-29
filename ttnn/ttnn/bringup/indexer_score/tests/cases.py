# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup indexer_score call a model makes (bringup-fork-tests skill). Append only; never
edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # Hy4 (preview) DSA indexer, last 5120-token prefill chunk of a 56320-token context on the 2x2 mesh: SP=2 over
        # axis 0 (the ring, cluster_axis 0) x TP=2 over axis 1 (query rows block-cyclic sub-sharded, seq_subshard_axis
        # 1); the index-key cache is striped over all 4 chips (block_cyclic_cache_tp_sharded, KV dedup) and k_local is
        # an SP row's TP-inner gather of it. HiFi4 + fp32 DEST (the fork's option), work unit q 64 x k 32.
        "id": "hy4_preview_d_p-2x2-ring-sp2tp2-q1280-t56320-h32-d128-fp32dest",
        "model": "hy4_preview_d_p",
        "task": "O.1",
        "sig": "9d64887fca",
        "op": "ring_indexer_score_dsa",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # q [1, 32, 1280, 128], k (the ring's full-T output buffer) [1, 1, 56320, 128], weights [1, 1, 1280, 32],
        # k_local [1, 1, 28160, 128]; all BFLOAT16 TILE DRAM interleaved, per device.
        "heads": 32,
        "head_dim": 128,
        "q_rows": 1280,
        "t": 56320,
        "k_local_rows": 28160,
        "dtype": "BFLOAT16",
        "layout": "TILE",
        "num_semaphores": 2,  # ag_multi_device_global_semaphore: 2 global semaphores on the full grid, value 0
        "cluster_axis": 0,
        "topology": "Linear",
        "num_links": 2,
        "chunk_start_idx": 51200,
        "kv_len": 56320,
        "seq_subshard_axis": 1,
        "block_cyclic_sp_axis": 0,
        "block_cyclic_chunk_local": 2560,
        "block_cyclic_cache_tp_sharded": True,
        "program_config": {"q_chunk_size": 64, "k_chunk_size": 32, "head_group_size": 0},
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        # q, k ~ N(0, 1); gates ~ N(0, 1) / 64 (the model folds 32^-0.5 * 128^-0.5 = 1/64 into them; some negative).
        # The ring's k buffer starts as random garbage (the model shares one buffer across layers, so it holds stale
        # keys): the op must fill every slab it reads.
        "gate_scale": 1.0 / 64,
        "seed": 0,
        # vs float32 on the same bf16 inputs, causal columns. Measured (seed 0, 4 chips, 66-71M scores each): pcc
        # 0.9999985, rel 0.00186 (the bf16 output rounding; the tests/unit/test_fp32_dest.py limit is 0.004). Checked by
        # hand: an output scaled by 1.01 fails, and so does k_local in natural (not block-cyclic) order.
        "pcc": 0.9999,
        "rel": 0.004,
    },
    {
        # GLM-5.3-Flash DSA indexer (tt/indexer.py, ring path, "mesh" geometry) on the 2x2 mesh, last 5120-token chunk
        # of a 56320-token prefill: pooled keys (key_compression_ratio 4), one full-mesh snake ring (cluster_axis None,
        # Ring). Chip d (row-major) scores its own 1280 query tokens (chunk start + d * 1280) and holds, per chunk, a
        # 320-pool stripe (block_cyclic_chunk_local 1280 tokens / 4) of the block-cyclic pooled-key cache: k_local
        # [5120, 128] (16 chunks' stripes, the 81920-token worst case); the ring's k buffer is [20480, 128]
        # (tt_ccl.get_indexer_ring_k_buffer: k_local rows x 4 chips). kv_len = (51200 + 5120) / 4 = 14080 pools.
        "id": "glm53_flash_d_p-2x2-ring-fullmesh-keystride4-q1280-t20480-kv14080-h32-d128-fp32dest",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "ef030f7670",
        "op": "ring_indexer_score_dsa",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # q [1, 32, 1280, 128], k [1, 1, 20480, 128], weights [1, 1, 1280, 32], k_local [1, 1, 5120, 128]; all
        # BFLOAT16 TILE DRAM interleaved, per device.
        "heads": 32,
        "head_dim": 128,
        "q_rows": 1280,
        "t": 20480,
        "k_local_rows": 5120,
        "dtype": "BFLOAT16",
        "layout": "TILE",
        "num_semaphores": 2,
        "cluster_axis": None,
        "topology": "Ring",
        "num_links": 1,
        "key_compression_ratio": 4,
        "chunk_start_idx": 51200,
        "kv_len": 14080,
        "block_cyclic_chunk_local": 1280,
        "program_config": {"q_chunk_size": 64, "k_chunk_size": 32, "head_group_size": 0},
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        # q, k ~ N(0, 1); gates ~ N(0, 1) / 64 (the model folds 32^-0.5 * 128^-0.5 = 1/64 into them). The ring's k
        # buffer starts as random garbage (the model shares one buffer across layers).
        "gate_scale": 1.0 / 64,
        "seed": 0,
        # vs float32 on the same bf16 inputs, visible columns [0, kv_len). Measured (seed 0, 4 chips, 16.6-17.8M
        # visible scores each): pcc 0.9999985, rel 0.00186 (the bf16 output rounding); limits as the hy4 case. Checked
        # by hand: an output scaled by 1.01 fails (rel 0.0095).
        "pcc": 0.9999,
        "rel": 0.004,
    },
]


# deepseek_v41_flash_d_p O.1 (scaled 2x2 run, ladder rung last: the last 2048-token chunk, start 53248, of a 55296-token
# prefill; tt/indexer.py): ring_indexer_score_dsa over the SP ring (axis 0, Linear, 2 links), the TP query sub-shard on
# axis 1, natural query blocks (device (d, c) scores tokens start + d W + c Sq + [0, Sq), W 1024, Sq 512) over the
# block-cyclic index-key cache (SP-sharded, TP-replicated: key t on SP rank (r t / W) % SP), 32 heads x 128, HiFi2 +
# fp32 DEST, k_chunk 32. Ratio 2: the compressed index keys (T = 27648); ratio 1 (T = 55296) with the block epilogue
# (block_keys 8): the candidate layer's block max (block_max_out + block_max_pin) and the re-index layers' candidate
# restriction (block_bias). Run by test_ring_indexer_score_dsa_ds41. Limits: future / masked columns -inf and visible
# ones finite (exact); visible scores within noise_limit (bf16 q / k / gates, bf16 out) with the FPU's operand bits at
# the call's fidelity (reference.dsa_score(fidelity=)); the block max bit-exact vs the op's own bf16 scores.
_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "O.1",
    "op": "ring_indexer_score_dsa",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576, "fabric_payload_bytes": 10880},
    "heads": 32,
    "head_dim": 128,
    "q_rows": 512,
    "chunk": 2048,
    "dtype": "BFLOAT16",
    "layout": "TILE",
    "num_semaphores": 2,
    "cluster_axis": 0,
    "topology": "Linear",
    "num_links": 2,
    "chunk_start_idx": 53248,
    "seq_subshard_axis": 1,
    "block_cyclic_sp_axis": 0,
    "block_cyclic_chunk_local": 1024,
    "natural_query_blocks": True,
    "program_config": {"q_chunk_size": 64, "k_chunk_size": 32, "head_group_size": 0},
    "compute_kernel_config": {
        "math_fidelity": "HiFi2",
        "math_approx_mode": False,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "dst_full_sync_en": False,
    },
    "gate_scale": 1.0 / 64,
    "noise": {"inputs": ["bf16", "bf16", "bf16"], "out": "bf16"},
}

CASES += [
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-ring-natural-ratio2-q512-t27648-h32-d128-hifi2",
        "sig": "377a477700",
        "key_compression_ratio": 2,
        "t": 27648,
        "k_local_rows": 13824,
        "kv_len": 27648,
        "seed": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-ring-natural-ratio1-q512-t55296-h32-d128-hifi2-block-bias-bk8",
        "sig": "f064b9bda3",
        "key_compression_ratio": 1,
        "t": 55296,
        "k_local_rows": 27648,
        "kv_len": 55296,
        "seed": 2,
        "block_keys": 8,
        "block_bias": {"shape": [1, 1, 512, 6912], "dtype": "BFLOAT16", "layout": "TILE"},
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-ring-natural-ratio1-q512-t55296-h32-d128-hifi2-block-max-pin-bk8",
        "sig": "f5914547b9",
        "key_compression_ratio": 1,
        "t": 55296,
        "k_local_rows": 27648,
        "kv_len": 55296,
        "seed": 3,
        "block_keys": 8,
        "block_max_out": {"shape": [1, 1, 512, 6912], "dtype": "FLOAT32", "layout": "ROW_MAJOR"},
        "block_max_pin": True,
    },
]

# deepseek_v41_flash_d_p lowbit (2x2, DS41-150): the same geometry with the index query and keys as bfloat4_b (the
# reference's fp4 width; INDEX_Q_FORMAT / INDEX_K_FORMAT=bfp4): inputs on the host packer's bfloat4_b grid, the ring
# gathering bfloat4_b key tiles. Limits as above, the noise floor's inputs at bfp4 (the values are on the grid).
CASES += [
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-ring-natural-ratio2-q512-t27648-h32-d128-hifi2-qk-bfp4",
        "sig": "lowbit-qk-bfp4-r2",
        "key_compression_ratio": 2,
        "t": 27648,
        "k_local_rows": 13824,
        "kv_len": 27648,
        "seed": 4,
        "qk_dtype": "bfloat4_b",
        "noise": {"inputs": ["bfp4", "bfp4", "bf16"], "out": "bf16"},
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-ring-natural-ratio1-q512-t55296-h32-d128-hifi2-qk-bfp4",
        "sig": "lowbit-qk-bfp4-r1",
        "key_compression_ratio": 1,
        "t": 55296,
        "k_local_rows": 27648,
        "kv_len": 55296,
        "seed": 5,
        "qk_dtype": "bfloat4_b",
        "noise": {"inputs": ["bfp4", "bfp4", "bf16"], "out": "bf16"},
    },
]
