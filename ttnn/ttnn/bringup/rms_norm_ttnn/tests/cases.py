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
        # GLM-5.3-Flash kv_a_layernorm on the 512-wide latent of all 5120 chunk rows (every chip), fp32 in and out (the
        # kv_a projection's fp32 output), bf16 weight, 2x2 mesh.
        "id": "glm53_flash_d_p-2x2-s5120-h512-fp32-w-eps1e-5",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "693a2b7a58",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 5120, 512] FLOAT32 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 5120, 512], "dtype": "FLOAT32", "layout": "TILE"},
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
        # 0.0094-0.0131, max rel 0.0022 (the FPU reads fp32 operands at TF32 precision). Limits atol 0.002 + rtol 0.004;
        # a 1.01 output scale fails them (checked by hand).
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
]
