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
]
