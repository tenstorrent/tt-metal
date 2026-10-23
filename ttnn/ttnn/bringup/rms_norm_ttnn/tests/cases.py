# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.rms_norm call a model makes (bringup-fork-tests skill). Append only; never edit
or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # MiMo-V2.6 input / post-attention RMSNorm on a 5120-token prefill chunk, replicated on a 1x4 mesh.
        "id": "mimo_v2_6_d_p-1x4-s5120-h4096-bf16-w-eps1e-6",
        "model": "mimo_v2_6_d_p",
        "task": "O.1",
        "sig": "8d29c1ff0f",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 5120, 4096] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 5120, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        # weight [1, 1, 1, 4096] bf16 TILE DRAM interleaved
        "weight": {"shape": [1, 1, 1, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "epsilon": 1e-06,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # measured (seed 0, 4 devices): pcc 0.9999972, max abs err 0.063, max rel err 0.0092 (bf16 output
        # rounding); atol 0.005 + rtol 0.008 passes, and a 1.01 scale of the output fails it (checked by hand)
        "pcc": 0.9999,
        "atol": 0.005,
        "rtol": 0.008,
    },
    {
        # MiMo-V2.6 fused residual add + RMSNorm (post-attention / next-layer input norm) on a 5120-token prefill
        # chunk, 1x4 mesh: t = x + residual is returned too (return_residual_sum), both DRAM interleaved.
        "id": "mimo_v2_6_d_p-1x4-s5120-h4096-bf16-w-res-sum-eps1e-6",
        "model": "mimo_v2_6_d_p",
        "task": "O.1",
        "sig": "952cfae417",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input and residual_input_tensor [1, 1, 5120, 4096] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 5120, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "residual": {"shape": [1, 1, 5120, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        # weight [1, 1, 1, 4096] bf16 TILE DRAM interleaved
        "weight": {"shape": [1, 1, 1, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "return_residual_sum": True,  # residual_sum_memory_config = DRAM interleaved
        "epsilon": 1e-06,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 1,
        # y: PCC + atol/rtol vs the float64 reference on t = x + residual (the op's statistics use the unrounded sum; a
        # reference on the bf16-rounded t is worse, max rel 0.015). Measured (seeds 1-3, 4 devices): pcc 0.9999959,
        # max abs err 0.087, max rel err 0.0112; 0 elements outside atol 0.005 + rtol 0.010, limit set at rtol 0.012.
        # t: bit-exact vs ttnn.add(x, residual) on the same device (the op's contract,
        # tests/unit/test_rms_norm_ttnn_residual_output.py). Checked by hand: a 1.01 scale of y fails, and one
        # corrupted element of t fails.
        "pcc": 0.9999,
        "atol": 0.005,
        "rtol": 0.012,
    },
    {
        # MiMo-V2.6 input / post-attention RMSNorm on a 5120-token prefill chunk, replicated on a 2x2 mesh.
        "id": "mimo_v2_6_d_p_2x2-2x2-s5120-h4096-bf16-w-eps1e-6",
        "model": "mimo_v2_6_d_p_2x2",
        "task": "O.1",
        "sig": "fdbf629524",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 5120, 4096] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 5120, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        # weight [1, 1, 1, 4096] bf16 TILE DRAM interleaved
        "weight": {"shape": [1, 1, 1, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "epsilon": 1e-06,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # Same op arguments as the 1x4 case, on 4 chips of a 2x2 mesh; limits as the 1x4 case. Measured (seed 0, 4 chips): pcc
        # 0.9999972, max abs err 0.054, max rel err 0.0093.
        "pcc": 0.9999,
        "atol": 0.005,
        "rtol": 0.008,
    },
    {
        # MiMo-V2.6 fused residual add + RMSNorm on a 5120-token prefill chunk, 2x2 mesh: t = x + residual is
        # returned too (return_residual_sum), both DRAM interleaved.
        "id": "mimo_v2_6_d_p_2x2-2x2-s5120-h4096-bf16-w-res-sum-eps1e-6",
        "model": "mimo_v2_6_d_p_2x2",
        "task": "O.1",
        "sig": "ab56dee4d6",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "input": {"shape": [1, 1, 5120, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "residual": {"shape": [1, 1, 5120, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "weight": {"shape": [1, 1, 1, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "return_residual_sum": True,  # residual_sum_memory_config = DRAM interleaved
        "epsilon": 1e-06,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 1,
        # y vs the float64 reference on t = x + residual; t bit-exact vs ttnn.add. Measured (seed 1, 4
        # chips): pcc 0.9999985, max abs err 0.044, max rel err 0.0062; limits as the 1x4 case.
        "pcc": 0.9999,
        "atol": 0.005,
        "rtol": 0.012,
    },
    {
        # GLM-5.3-Flash q_a_layernorm on this chip's 1280 rows of a 5120-token chunk (split residual layout), fp32 in and
        # out (the q_a projection's fp32 output), bf16 weight, 2x2 mesh.
        "id": "glm53_flash_d_p-2x2-s1280-h1536-fp32-w-eps1e-5",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "3689df566e",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 1536] FLOAT32 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 1536], "dtype": "FLOAT32", "layout": "TILE"},
        # weight [1, 1, 1, 1536] bf16 TILE DRAM interleaved
        "weight": {"shape": [1, 1, 1, 1536], "dtype": "BFLOAT16", "layout": "TILE"},
        "epsilon": 1e-05,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # float64 reference on the same fp32 x and bf16 w. Measured (seed 0, 4 chips): pcc 0.9999999, max abs err
        # 0.0098-0.0111, max rel 0.0022 (the FPU reads fp32 operands at TF32 precision). Limits atol 0.002 + rtol 0.004;
        # a 1.01 output scale fails them (checked by hand).
        "pcc": 0.99999,
        "atol": 0.002,
        "rtol": 0.004,
    },
    {
        # GLM-5.3-Flash kv_a_layernorm on the 512-wide latent of this chip's 1280 rows of a 5120-token chunk (the
        # block-cyclic striped latent cache: each chip normalizes only its own rows), fp32 in and out (the kv_a
        # projection's fp32 output), bf16 weight, 2x2 mesh.
        "id": "glm53_flash_d_p-2x2-s1280-h512-fp32-w-eps1e-5",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "77f466eb2c",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 512] FLOAT32 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 512], "dtype": "FLOAT32", "layout": "TILE"},
        # weight [1, 1, 1, 512] bf16 TILE DRAM interleaved
        "weight": {"shape": [1, 1, 1, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "epsilon": 1e-05,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # float64 reference on the same fp32 x and bf16 w. Measured (seed 0, 4 chips): pcc 0.9999999, max abs err
        # 0.0092-0.0101, max rel 0.0021-0.0022 (TF32 operand reads). Limits atol 0.002 + rtol 0.004 (as the other
        # glm fp32 case); a 1.01 output scale fails them (checked by hand: 441586 elements out on chip 0).
        "pcc": 0.99999,
        "atol": 0.002,
        "rtol": 0.004,
    },
    {
        # GLM-5.3-Flash input / post-attention RMSNorm (and the other hidden-size norms) on this chip's 1280 rows of a
        # 5120-token chunk (split residual layout), bf16, 2x2 mesh.
        "id": "glm53_flash_d_p-2x2-s1280-h4096-bf16-w-eps1e-5",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "6fa50c3954",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 4096] BFLOAT16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        # weight [1, 1, 1, 4096] bf16 TILE DRAM interleaved
        "weight": {"shape": [1, 1, 1, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "epsilon": 1e-05,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # Measured (seed 0, 4 chips): pcc 0.9999986, max abs err 0.034-0.037, max rel 0.0046 (bf16 output rounding);
        # limits as the MiMo bf16 cases, which a 1.01 output scale fails.
        "pcc": 0.9999,
        "atol": 0.005,
        "rtol": 0.008,
    },
    {
        # Hy4 q_a_layernorm (tt/q_a.py, q_lora_rank 2048) on the 2560 rows of a 2x2 chip (5120-token chunk), fp32 in /
        # out, fp32 row-major weight.
        "id": "hy4_preview_d_p-2x2-s2560-h2048-fp32-rmw-eps1e-6",
        "model": "hy4_preview_d_p",
        "task": "O.1",
        "sig": "16501544dc",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 2560, 2048] fp32 TILE DRAM interleaved (per device); fp32 random values (not bf16-rounded)
        "input": {"shape": [1, 1, 2560, 2048], "dtype": "FLOAT32", "layout": "TILE"},
        # weight [1, 1, 64, 32] fp32 ROW_MAJOR DRAM interleaved: the 2048 channels as rows of 32 (replicated)
        "weight": {"shape": [1, 1, 64, 32], "dtype": "FLOAT32", "layout": "ROW_MAJOR"},
        "epsilon": 1e-06,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # vs the float64 reference on the same fp32 x / w. Measured (seed as above, 4 chips): pcc 0.9999998, max abs err
        # 0.024, max rel err 0.0030 (the FPU's fp32 multiply); limits at 2x the measured rel error. A 1.01 scale of the
        # output fails (checked by hand).
        "pcc": 0.99999,
        "atol": 0.002,
        "rtol": 0.006,
    },
    {
        # Hy4 kv_a_layernorm (kv_lora_rank 512) on the 2560 rows of a 2x2 chip, fp32 in / out, fp32 row-major weight.
        "id": "hy4_preview_d_p-2x2-s2560-h512-fp32-rmw-eps1e-6",
        "model": "hy4_preview_d_p",
        "task": "O.1",
        "sig": "31912e9ee9",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 2560, 512] fp32 TILE DRAM interleaved (per device); fp32 random values (not bf16-rounded)
        "input": {"shape": [1, 1, 2560, 512], "dtype": "FLOAT32", "layout": "TILE"},
        # weight [1, 1, 16, 32] fp32 ROW_MAJOR DRAM interleaved: the 512 channels as rows of 32 (replicated)
        "weight": {"shape": [1, 1, 16, 32], "dtype": "FLOAT32", "layout": "ROW_MAJOR"},
        "epsilon": 1e-06,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 1,
        # vs the float64 reference on the same fp32 x / w. Measured (seed as above, 4 chips): pcc 0.9999998, max abs err
        # 0.015, max rel err 0.0030 (the FPU's fp32 multiply); limits at 2x the measured rel error. A 1.01 scale of the
        # output fails (checked by hand).
        "pcc": 0.99999,
        "atol": 0.002,
        "rtol": 0.006,
    },
    {
        # Hy4 hidden RMSNorm (tt/norm.py:TtGatheredRmsNorm, 6144, after the TP all_gather: ffn_norm, final norm) on
        # the 2560 rows of a 2x2 chip, fp32 in / out, fp32 row-major weight, eps 1e-5.
        "id": "hy4_preview_d_p-2x2-s2560-h6144-fp32-rmw-eps1e-5",
        "model": "hy4_preview_d_p",
        "task": "O.1",
        "sig": "d24c5a2de5",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 2560, 6144] fp32 TILE DRAM interleaved (per device); fp32 random values (not bf16-rounded)
        "input": {"shape": [1, 1, 2560, 6144], "dtype": "FLOAT32", "layout": "TILE"},
        # weight [1, 1, 192, 32] fp32 ROW_MAJOR DRAM interleaved: the 6144 channels as rows of 32 (replicated)
        "weight": {"shape": [1, 1, 192, 32], "dtype": "FLOAT32", "layout": "ROW_MAJOR"},
        "epsilon": 1e-05,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 2,
        # vs the float64 reference on the same fp32 x / w. Measured (seed as above, 4 chips): pcc 0.9999998, max abs err
        # 0.020, max rel err 0.0029 (the FPU's fp32 multiply); limits at 2x the measured rel error. A 1.01 scale of the
        # output fails (checked by hand).
        "pcc": 0.99999,
        "atol": 0.002,
        "rtol": 0.006,
    },
    {
        # Xing4.0 kv_a_layernorm (kv_lora_rank 512) on the 1280 rows of a 4x2 chip, fp32 in / out, fp32 row-major weight
        # ([1, 1, 16, 32]: the 512 channels as rows of 32, replicated); DRAM interleaved.
        "id": "xing40_a4b_d_p-4x2-s1280-h512-fp32-rmw-eps1e-6",
        "model": "xing40_a4b_d_p",
        "task": "O.1",
        "sig": "0aa85ca05d",
        "mesh": [4, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "input": {"shape": [1, 1, 1280, 512], "dtype": "FLOAT32", "layout": "TILE"},
        "weight": {"shape": [1, 1, 16, 32], "dtype": "FLOAT32", "layout": "ROW_MAJOR"},
        "epsilon": 1e-06,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 0,
        # vs the float64 reference on the same fp32 x / w. Measured (8 chips): pcc 0.9999998, max abs err <= 0.022, max
        # rel err <= 0.0030; limits as the hy4 fp32 row-major-weight cases (2x the measured rel error).
        "pcc": 0.99999,
        "atol": 0.002,
        "rtol": 0.006,
    },
    {
        # Xing4.0 q_a_layernorm (q_lora_rank 768) on the 1280 rows of a 4x2 chip, fp32 in / out, fp32 row-major weight
        # ([1, 1, 24, 32]: the 768 channels as rows of 32, replicated); DRAM interleaved.
        "id": "xing40_a4b_d_p-4x2-s1280-h768-fp32-rmw-eps1e-6",
        "model": "xing40_a4b_d_p",
        "task": "O.1",
        "sig": "c83e5f680d",
        "mesh": [4, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "input": {"shape": [1, 1, 1280, 768], "dtype": "FLOAT32", "layout": "TILE"},
        "weight": {"shape": [1, 1, 24, 32], "dtype": "FLOAT32", "layout": "ROW_MAJOR"},
        "epsilon": 1e-06,
        "compute_kernel_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
            "dst_full_sync_en": False,
        },
        "seed": 1,
        # vs the float64 reference on the same fp32 x / w. Measured (8 chips): pcc 0.9999998, max abs err <= 0.022, max
        # rel err <= 0.0030; limits as the hy4 fp32 row-major-weight cases (2x the measured rel error).
        "pcc": 0.99999,
        "atol": 0.002,
        "rtol": 0.006,
    },
]


# deepseek_v41_flash_d_p O.1 (scaled 2x2 run, ladder rung last: 1024 rows a chip; tt/norm.py, NORM_IMPL on): every
# norm through the fork, HiFi4 + fp32 DEST, fp32 TILE weights (bf16-valued, as the checkpoint's), epsilon 1e-20 (the
# GPU code's), DRAM interleaved. Run by test_rms_norm_ttnn_ds41 (the split program's options: stats_out, partial_stats /
# stats_width, input_window, dtype). Limits: noise_limit (F139) of the float64 reference at the declared formats
# (inputs, out), never a measured number; stats_out: the row sums of the partials vs float64 sum x^2.
_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "O.1",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
    "compute_kernel_config": {
        "math_fidelity": "HiFi4",
        "math_approx_mode": False,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "dst_full_sync_en": False,
    },
    "seed": 0,
}

CASES += [
    {
        # compressor norm (tt/compressor.py), fp32 rows -> bf16 (dtype=), the default program's statistic over the full 512 width
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-h512-fp32-to-bf16",
        "sig": "516a31a79d",
        "input": {"shape": [1, 1, 1024, 512], "dtype": "FLOAT32", "layout": "TILE"},
        "weight": {"shape": [1, 1, 1, 512], "dtype": "FLOAT32", "layout": "TILE"},
        "epsilon": 1e-20,
        "dtype": "BFLOAT16",
        "noise": {"inputs": ["fp32", "fp32"], "out": "bf16"},
    },
    {
        # compressor norm on the 512 rows of the ratio-4 path, fp32 rows -> bf16 (dtype=)
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s512-h512-fp32-to-bf16",
        "sig": "5f7e0b4ad1",
        "input": {"shape": [1, 1, 512, 512], "dtype": "FLOAT32", "layout": "TILE"},
        "weight": {"shape": [1, 1, 1, 512], "dtype": "FLOAT32", "layout": "TILE"},
        "epsilon": 1e-20,
        "dtype": "BFLOAT16",
        "noise": {"inputs": ["fp32", "fp32"], "out": "bf16"},
    },
    {
        # k_norm on the indexer keys, bf16 in / out
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-h128-bf16",
        "sig": "68a871de3d",
        "input": {"shape": [1, 1, 1024, 128], "dtype": "BFLOAT16", "layout": "TILE"},
        "weight": {"shape": [1, 1, 1, 128], "dtype": "FLOAT32", "layout": "TILE"},
        "epsilon": 1e-20,
        "noise": {"inputs": ["bf16", "fp32"], "out": "bf16"},
    },
    {
        # attn_norm / ffn_norm apply step: this chip's 2560 of 5120 hidden columns, normalized by the whole row's statistic (partial_stats = the all-gathered stats_out partials of both TP chips, stats_width 5120), bf16 out
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-h2560-bf16-partial-stats5120",
        "sig": "73aa2d360c",
        "input": {"shape": [1, 1, 1024, 2560], "dtype": "BFLOAT16", "layout": "TILE"},
        "weight": {"shape": [1, 1, 1, 2560], "dtype": "FLOAT32", "layout": "TILE"},
        "epsilon": 1e-20,
        "partial_stats": {"shape": [1, 1, 1024, 128], "dtype": "FLOAT32", "layout": "TILE"},
        "stats_width": 5120,
        "noise": {"inputs": ["bf16", "fp32", "fp32"], "out": "bf16"},
    },
    {
        # attn_norm / ffn_norm statistic step: the partial sums of squares of this chip's 2560 hidden columns (stats_out, fp32 [.., 32 B])
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-h2560-bf16-stats-out",
        "sig": "826864a291",
        "input": {"shape": [1, 1, 1024, 2560], "dtype": "BFLOAT16", "layout": "TILE"},
        "weight": None,
        "epsilon": None,
        "stats_out": True,
        "noise": {"inputs": ["bf16"], "out": "fp32"},
    },
    {
        # kv_norm on the fused [wq_a | wkv] output: columns [1280, 1792) read in place (input_window), bf16 out
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-h1792-bf16-window1280-1792",
        "sig": "9166a4be85",
        "input": {"shape": [1, 1, 1024, 1792], "dtype": "BFLOAT16", "layout": "TILE"},
        "weight": {"shape": [1, 1, 1, 512], "dtype": "FLOAT32", "layout": "TILE"},
        "epsilon": 1e-20,
        "input_window": [1280, 1792],
        "noise": {"inputs": ["bf16", "fp32"], "out": "bf16"},
    },
    {
        # k_norm on the 512-row indexer path, bf16 in / out
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s512-h128-bf16",
        "sig": "a4b71ffddb",
        "input": {"shape": [1, 1, 512, 128], "dtype": "BFLOAT16", "layout": "TILE"},
        "weight": {"shape": [1, 1, 1, 128], "dtype": "FLOAT32", "layout": "TILE"},
        "epsilon": 1e-20,
        "noise": {"inputs": ["bf16", "fp32"], "out": "bf16"},
    },
    {
        # q_norm on the fused [wq_a | wkv] output: columns [0, 1280) read in place (input_window), bf16 out
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-h1792-bf16-window0-1280",
        "sig": "f3058f954f",
        "input": {"shape": [1, 1, 1024, 1792], "dtype": "BFLOAT16", "layout": "TILE"},
        "weight": {"shape": [1, 1, 1, 1280], "dtype": "FLOAT32", "layout": "TILE"},
        "epsilon": 1e-20,
        "input_window": [0, 1280],
        "noise": {"inputs": ["bf16", "fp32"], "out": "bf16"},
    },
]
