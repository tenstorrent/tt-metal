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
]
