# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.mhc_post call a model makes (bringup-fork-tests skill). Append only; never edit
or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # GLM-5.3-Flash mHC post (attn / ffn residual) on a 5120-token chunk, split layout: each chip its 1280 rows.
        "id": "glm53_flash_d_p-2x2-s1280-c4096-n4-bf16",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "eee7cc7ddf",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input F [1, 1, 1280, 4096] bf16, residual X [1, 1, 1280, 16384] bf16, post [1, 1, 1280, 4] fp32,
        # comb [1, 1, 1280, 16] fp32; all TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "residual": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        "post": {"shape": [1, 1, 1280, 4], "dtype": "FLOAT32", "layout": "TILE"},
        "comb": {"shape": [1, 1, 1280, 16], "dtype": "FLOAT32", "layout": "TILE"},
        "seed": 0,
        # measured (seed 0, 4 devices): rel L2 0.00166 (bf16 output rounding), pcc 0.9999986; a 1.01 scale of the output
        # (rel 0.01) fails this limit
        "pcc": 0.9999,
        "max_rel": 0.004,
    },
]
