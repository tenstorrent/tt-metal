# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model cases of the sdpa fork (bring-up task O.1): one dict per captured ttnn.bringup call, every model's, appended.

ring_joint cases ("op": "ring_joint"): ttnn.bringup.ring_joint_scaled_dot_product_attention on the metadata (slot_id /
kv_actual_isl_tensor) chunked path over a served, block-cyclic, ND-sharded bfloat8_b KV cache. Per chip: q
[1, heads, chunk // sp, d] bf16, cache [slots, kv_heads, local_seq, d] (global position g on SP row (g // W) % sp at
local row (g // chunk) * W + g % W, W = chunk // sp), persistent gather buffers [1, kv_heads, gather_rows, d].
kv_actual / slot: the A inputs; b_kv_actual / b_slot: the B inputs of the determinism check (built with seed + 1).
"""

CASES = [
    {
        "id": "mimo_v26_flash_rl-ring_joint-sliding-w128-sink-exact_exp",
        "model": "mimo_v26_flash_rl",
        "task": "O.1",
        "sig": "a3c4b03e41",
        "op": "ring_joint",
        "mesh": [8, 4],
        "fabric_config": "FABRIC_1D_RING",
        "cluster_axis": 0,
        "heads": 16,  # per chip (TP column c: global q heads 16 c ..)
        "kv_heads": 2,  # per chip
        "chunk": 5120,  # global; 640 q rows per chip
        "slots": 2,
        "local_seq": 7040,
        "k_dim": 192,
        "v_dim": 192,
        "gather_rows": 256,  # two predecessor-halo slots of 128 rows
        "logical_n": 56320,
        "scale": 0.07216878364870322,
        "sliding_window_size": 128,
        "attention_sink": True,  # [1, heads, 1, 1] bf16 per chip
        "q_chunk": 128,
        "k_chunk": 128,
        "segmented_accumulation": False,
        "sdpa_grid": [11, 10],
        "ccl_core_grid_offset": [11, 0],
        "num_links": 2,
        "topology": "Linear",
        "use_column_major_ccl": True,
        "math_fidelity": "HiFi4",
        "math_approx_mode": False,
        "fp32_dest_acc_en": False,
        "packer_l1_acc": False,
        "exact_exp": True,
        "kv_cache_num_layers": 1,
        "kv_cache_layer_idx": 0,
        "seed": 202,
        # A: a mid-slab start (SP row 2 holds two runs, so both halo slots are used); B: chunk 0 in the other slot
        "kv_actual": 6496,
        "slot": 1,
        "b_kv_actual": 0,
        "b_slot": 0,
        "q_scale": 1.0,
        # the model discards this call's output and reads the gather buffers' halo rows: those before kv_actual are
        # checked exactly, the output against reference.py
        "check_halo": True,
        # measured on seeds 202 / 303 / 404 / 505: pcc >= 0.99971, rel 0.0148-0.0151, norm ratio 1.0044-1.0049 (the
        # bf16-dest streaming softmax with a sink biases the norm up), worst (head, row) rel 0.056-0.064; the output
        # x 1.01 fails rel and the norm ratio
        "pcc": 0.999,
        "rel": 0.018,
        "norm_ratio": [0.997, 1.008],
        "row_rel": 0.08,
    },
    {
        "id": "mimo_v26_flash_rl-ring_joint-full-causal-segmented-k320-exact_exp",
        "model": "mimo_v26_flash_rl",
        "task": "O.1",
        "sig": "ac3ce65869",
        "op": "ring_joint",
        "mesh": [8, 4],
        "fabric_config": "FABRIC_1D_RING",
        "cluster_axis": 0,
        "heads": 16,
        "kv_heads": 1,
        "chunk": 5120,
        "slots": 2,
        "local_seq": 7040,
        "k_dim": 192,
        "v_dim": 128,
        "gather_rows": 56320,  # the whole gathered prefix
        "logical_n": 56320,
        "scale": 1.0,  # the model folds 192^-0.5 into q
        "sliding_window_size": None,
        "attention_sink": False,
        "q_chunk": 128,
        "k_chunk": 320,
        "segmented_accumulation": True,
        "sdpa_grid": [11, 10],
        "ccl_core_grid_offset": [11, 0],
        "num_links": 2,
        "topology": "Linear",
        "use_column_major_ccl": True,
        "math_fidelity": "HiFi4",
        "math_approx_mode": False,
        "fp32_dest_acc_en": False,
        "packer_l1_acc": False,
        "exact_exp": True,
        "kv_cache_num_layers": 1,
        "kv_cache_layer_idx": 0,
        "seed": 101,
        # A: a deep mid-slab start (prefix 25632, SP row 0 holds two runs); B: an aligned start in the other slot
        "kv_actual": 25632,
        "slot": 1,
        "b_kv_actual": 10240,
        "b_slot": 0,
        "q_scale": 192**-0.5 * 2,  # scores ~ N(0, 4): softmax neither uniform nor one-hot
        "check_halo": False,
        # measured on seeds 101 / 303 / 404 / 505: pcc >= 0.99995, rel 0.0065-0.0066, norm ratio 1.0001, worst (head,
        # row) rel 0.16-0.18 (a near-cancelling row); the output x 1.01 fails rel and the norm ratio
        "pcc": 0.9995,
        "rel": 0.009,
        "norm_ratio": [0.997, 1.003],
        "row_rel": 0.22,
    },
]
