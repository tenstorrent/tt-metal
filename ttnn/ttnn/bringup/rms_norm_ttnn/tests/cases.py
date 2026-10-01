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
