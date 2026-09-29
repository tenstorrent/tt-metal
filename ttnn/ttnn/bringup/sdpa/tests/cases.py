# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup SDPA call a model makes (bringup-fork-tests skill). Append only; never edit or
loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json); every tensor bf16 TILE DRAM
interleaved unless stated. Program config grid [11, 10], max_cores_per_head_batch 16."""

CASES = [
    {
        # MiMo-V2.6 full-attention layer, chunked prefill of the last 5120-token chunk after a 51200-token prefix, on
        # a paged cache of 880 blocks x 64 (56320 tokens), 1 KV head per chip (GQA 16:1), K 192, V 128 (the fork's
        # narrow V). Config "A": HiFi2, fp32 dest off (streaming), approx exp, q512/k128. chunk_start_idx an int.
        "id": "mimo_v2_6_d_p-1x4-chunked-paged-q16x5120-kv1-prefix51200-k192-v128-hifi2-q512k128",
        "model": "mimo_v2_6_d_p",
        "task": "O.1",
        "sig": "239d54bae3",
        "op": "chunked_scaled_dot_product_attention",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "q": [1, 16, 5120, 192],
        "k": [880, 1, 64, 192],  # paged [blocks, nkv, block, D]
        "v": [880, 1, 64, 128],
        "page_table": [1, 880],  # INT32 ROW_MAJOR; the test fills it with a random permutation of the blocks
        "chunk_start_idx": 51200,
        "scale": 0.07216878235340118,  # 192^-0.5 rounded to fp32
        "is_causal": None,
        "sliding_window_size": None,
        "attention_sink": None,
        "compute_kernel_config": {
            "math_fidelity": "HiFi2",
            "math_approx_mode": False,
            "fp32_dest_acc_en": False,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "program_config": {"grid": [11, 10], "q_chunk_size": 512, "k_chunk_size": 128, "exp_approx_mode": True},
        "seed": 0,
        # Checks: bit-identical to the source op on V zero-padded to 192 (first 128 columns), and vs the float32
        # reference. Measured (seed 0, 4 devices): pcc 0.99916-0.99921, rel L2 0.090-0.092, max abs 0.006. Random
        # Q/K over a 56k-key context give near-uniform attention and a tiny output (|out| ~ 0.004), so bf16 / HiFi2 /
        # approx-exp error is large relative to it (the padded source op has the same error).
        "pcc": 0.998,
        "rel": 0.12,
    },
    {
        # MiMo-V2.6 sliding-window layer (window 128, per-head attention sink), 5120-token chunk with the 128-token
        # window tail prepended (S 5248), 2 KV heads per chip (GQA 8:1), K 192, V 128 (narrow V). SDPA scale is the
        # power of two 2^-4 (the model folds the rest into Q); the sink is stored pre-divided by it. Config "S":
        # HiFi4, fp32 dest off (streaming), exact exp, q128/k128.
        "id": "mimo_v2_6_d_p-1x4-causal-swa128-sink-q16x5248-kv2-k192-v128-hifi4-q128k128",
        "model": "mimo_v2_6_d_p",
        "task": "O.1",
        "sig": "3f237ff436",
        "op": "scaled_dot_product_attention",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "q": [1, 16, 5248, 192],
        "k": [1, 2, 5248, 192],
        "v": [1, 2, 5248, 128],
        "page_table": None,
        "chunk_start_idx": None,
        "scale": 0.0625,
        "is_causal": True,
        "sliding_window_size": 128,
        "attention_sink": [1, 16, 1, 1],  # the test draws the sink logit in [0, 3), stores logit / scale
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": False,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "program_config": {"grid": [11, 10], "q_chunk_size": 128, "k_chunk_size": 128, "exp_approx_mode": False},
        "seed": 1,
        # Checks: bit-identical to the source op on V zero-padded to 192, and vs the float32 reference. Measured
        # (seed 1, 4 devices): pcc 0.99974, rel L2 0.0232-0.0233, max abs 0.053.
        "pcc": 0.9995,
        "rel": 0.03,
    },
    {
        # MiMo-V2.6 full-attention layer on a 2x2 mesh: chunked prefill of the last 5120-token chunk after a
        # 51200-token prefix, paged cache of 880 blocks x 64, 1 KV head per chip (GQA 16:1), K 192, V 128 (narrow V).
        # HiFi2, fp32 dest off, approx exp, q512/k128 (perf pick P.1). chunk_start_idx an int.
        "id": "mimo_v2_6_d_p_2x2-2x2-chunked-paged-q16x5120-kv1-prefix51200-k192-v128-hifi2-q512k128",
        "model": "mimo_v2_6_d_p_2x2",
        "task": "O.1",
        "sig": "f34d0b9d1f",
        "op": "chunked_scaled_dot_product_attention",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "q": [1, 16, 5120, 192],
        "k": [880, 1, 64, 192],
        "v": [880, 1, 64, 128],
        "page_table": [1, 880],
        "chunk_start_idx": 51200,
        "scale": 0.07216878235340118,
        "is_causal": None,
        "sliding_window_size": None,
        "attention_sink": None,
        "compute_kernel_config": {
            "math_fidelity": "HiFi2",
            "math_approx_mode": False,
            "fp32_dest_acc_en": False,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "program_config": {"grid": [11, 10], "q_chunk_size": 512, "k_chunk_size": 128, "exp_approx_mode": True},
        "seed": 0,
        # Bit-identical to the source op on V padded to 192, and vs the float32 reference. Measured (seed 0,
        # 4 chips): pcc 0.99916-0.99921, rel L2 0.090-0.092, max abs 0.006; limits as the 1x4 case.
        "pcc": 0.998,
        "rel": 0.12,
    },
    {
        # MiMo-V2.6 sliding-window layer (window 128, per-head attention sink) on a 2x2 mesh, 5120-token chunk with
        # the 128-token window tail prepended (S 5248), 2 KV heads per chip, K 192, V 128. Scale 2^-4, sink stored
        # pre-divided by it. HiFi4, fp32 dest off, exact exp, q128/k128.
        "id": "mimo_v2_6_d_p_2x2-2x2-causal-swa128-sink-q16x5248-kv2-k192-v128-hifi4-q128k128",
        "model": "mimo_v2_6_d_p_2x2",
        "task": "O.1",
        "sig": "4caa547c27",
        "op": "scaled_dot_product_attention",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "q": [1, 16, 5248, 192],
        "k": [1, 2, 5248, 192],
        "v": [1, 2, 5248, 128],
        "page_table": None,
        "chunk_start_idx": None,
        "scale": 0.0625,
        "is_causal": True,
        "sliding_window_size": 128,
        "attention_sink": [1, 16, 1, 1],
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": False,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "program_config": {"grid": [11, 10], "q_chunk_size": 128, "k_chunk_size": 128, "exp_approx_mode": False},
        "seed": 1,
        # Bit-identical to the source op on V padded to 192, and vs the float32 reference. Measured (seed 1, 4
        # chips): pcc 0.99974, rel L2 0.0232-0.0233, max abs 0.053; limits as the 1x4 case.
        "pcc": 0.9995,
        "rel": 0.03,
    },
    {
        # GLM-5.3-Flash DSA sparse MLA (NoPE, absorbed kv_b) on a 2x2 mesh: the last 5120-token chunk of a 56320-token
        # context, chip d holding query rows 1280 d .. + 1279 of the chunk, all 64 heads; kv = the latent cache
        # [1, 1, 56320, 512] (K = V = its 512 columns), 2176 ids per row (2051 selected + 0xFFFFFFFF sentinels),
        # k_chunk 128 (17 chunks), scale 256^-0.5 = 1/16, HiFi4 + fp32 dest, high_precision (the fork's option).
        # q / kv / idx bf16 / bf16 / uint32, all ROW_MAJOR DRAM interleaved; output [1, 64, 1280, 512].
        "id": "glm53_flash_d_p-2x2-sparse-q64x1280-kv56320-k512-idx2176-hifi4-hp-kc128",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "d894c9aab4",
        "op": "sparse_sdpa",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "q": [1, 64, 1280, 512],
        "kv": [1, 1, 56320, 512],
        "indices": [1, 1, 1280, 2176],
        "v_dim": 512,
        "q_pos": 51200,  # position of chip 0's first query row
        "n_valid": 2051,
        "n_short_every": 16,  # every 16th row: 1 .. 2051 valid ids
        "kv_format": "BF16",
        "scale": 0.0625,
        "k_chunk_size": 128,
        "high_precision": True,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # vs the float32 reference on the same bf16 inputs. Measured (seed 0, 4 chips): pcc 0.999984, rel L2
        # 0.0057-0.0059, per (head, row) norm ratio [0.9867, 1.0139]. Every full row is off by ~0.0055: random q / kv
        # give near-uniform attention over 2051 ids (|out| ~ 1.9 vs ~ 22.6 for one kv row). The unit test's 0.0017
        # at this geometry is diluted by its exact 1-id rows (their norm dominates); with all rows at 2051 ids the
        # same op gives 0.0058 there too. A 1.01 output scale (rel ~ 0.0115, ratio max ~ 1.024) fails both limits.
        "pcc": 0.9999,
        "rel": 0.008,
        "ratio": [0.98, 1.02],
    },
]
