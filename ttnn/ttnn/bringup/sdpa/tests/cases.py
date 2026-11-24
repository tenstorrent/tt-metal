# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model cases of the sdpa fork: one dict per captured ttnn.bringup call (bring-up task O.1), every model's, appended.

Shapes are per device, as captured (fork_calls.json). Values are literal: a case must survive its model changing.
"""

CASES = [
    {
        # Kimi-K2.7 MLA chunked prefill, last ladder rung: chunk 10 (rows [51200, 56320)) of a 56320-row cache slot,
        # 16 heads per TP column, latent V = KV[:, :512], fp32 DEST + the fork's exact_score_exp.
        "id": "kimi_k27_code-ring_mla-chunk10of11-h16-q640-kv7040-fp32dest-exactexp",
        "model": "kimi_k27_code",
        "task": "O.1",
        "sig": "2d7687715d",
        "op": "ring_mla",
        "mesh": [8, 4],
        "fabric_config": "FABRIC_1D_RING",
        "sp_axis": 0,  # cluster_axis: the ring; Q and KV sequence-sharded over it
        "tp_axis": 1,  # Q heads split over it; KV replicated
        "q_shape": [1, 16, 640, 576],  # per device, bfloat16 TILE DRAM interleaved
        "kv_shape": [1, 1, 7040, 576],  # per device: 11 chunk slabs of 640 rows, block-cyclic
        "gather_shape": [1, 1, 56320, 576],  # persistent_output_buffer_kv, replicated
        "dtype": "bfloat16",
        "chunk_local": 640,  # rows per SP device per chunk (block-cyclic period)
        "head_dim_v": 512,
        "logical_n": 56320,
        "kv_actual_isl": 51200,
        "kv_cache_batch_idx": 0,
        "scale": 0.14467962580268923,
        "program_config": {
            "grid": [11, 10],
            "q_chunk_size": 128,
            "k_chunk_size": 64,
            "exp_approx_mode": False,
            "segmented_accumulation": True,
        },
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
        },
        "semaphore_grid": [12, 10],  # two global semaphores over the full compute grid
        "dim": 2,
        "num_links": 1,
        "topology": "Ring",
        "ccl_core_grid_offset": [11, 0],
        "use_column_major_ccl": True,
        "is_balanced": False,
        "exact_score_exp": True,
        "seed": 1234,
        # vs fp32 exact attention, seeds 1234..1238: pcc 0.999883, rel 0.0163 .. 0.0164 (bf16 inputs, output and
        # qk / sum CBs); the device output x1.01 measures rel 0.0218 and fails
        "pcc": 0.9998,
        "rel": 0.02,
    },
    {
        # hy4_preview attention (both attention layers), chunk 640 at the last ladder rung: per chip 64 heads x 160
        # query rows (heads -> sequence over TP), the slot's kvpe [0, 56320) in natural order, top-2048 ids
        "id": "hy4_preview-sparse_sdpa-h64-s160-t56320-k2048-sink-exact_exp",
        "model": "hy4_preview",
        "task": "O.1",
        "sig": "676f72c7a6",
        "op": "sparse_sdpa",
        "mesh": [8, 4],
        "fabric_config": "FABRIC_1D_RING",
        "q_shape": [1, 64, 160, 576],
        "kv_shape": [1, 1, 56320, 576],
        "indices_shape": [1, 1, 160, 2048],
        "sink_shape": [1, 1, 1, 64],
        "dtype": "bfloat16",
        "layout": "ROW_MAJOR",
        "memory": "DRAM_INTERLEAVED",
        "v_dim": 512,
        "kv_format": "BF16",
        "scale": 0.0625,
        "k_chunk_size": 128,
        "exact_exp": True,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 1234,
        # measured (seeds 1234/1235/1236, BH Galaxy): pcc 0.999993 / 0.999989 / 0.999909, rel 0.00699 / 0.00693 /
        # 0.00676 (the op's bf16 running output / row-sum); a 1 % scale of the output fails max_rel
        "pcc": 0.9995,
        "max_rel": 0.01,
    },
]
