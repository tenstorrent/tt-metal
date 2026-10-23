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
    {
        # Xing4.0 mHC residual (attn / ffn residual, tt/residual.py fused) on a 5120-token chunk of the 4x2 SP x TP
        # mesh: each chip its 1280 rows and 1792 of 3584 hidden columns; comb applied as stored (comb_transposed=False).
        # Per-device shapes; runs on the fixture's mesh (the per-chip math does not depend on the mesh shape).
        "id": "xing40_a4b_d_p-4x2-s1280-c1792-n4-fp32-comb",
        "model": "xing40_a4b_d_p",
        "task": "P.2",
        "sig": "f7215974bc",  # O.1: the captured call this case reproduces (fork_calls.json)
        "mesh": [4, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "input": {"shape": [1, 1, 1280, 1792], "dtype": "FLOAT32", "layout": "TILE"},
        "residual": {"shape": [1, 1, 1280, 7168], "dtype": "FLOAT32", "layout": "TILE"},
        "post": {"shape": [1, 1, 1280, 4], "dtype": "FLOAT32", "layout": "TILE"},
        "comb": {"shape": [1, 1, 1280, 16], "dtype": "FLOAT32", "layout": "TILE"},
        "comb_transposed": False,
        "seed": 1,
        # fp32 in / out: the op's measured fp32 rel RMS is ~5e-8 (changelog Phase 0)
        "pcc": 0.99999,
        "max_rel": 1e-5,
    },
]

# deepseek_v41_flash_d_p O.1 (scaled 2x2 run, ladder rung last: 1024 rows a chip, the chip's 2560 of 5120 hidden
# columns, n 4 bf16 streams; BF16_ACTIVATIONS, MHC_FUSED and GLUE_FOLD on). Run by test_mhc_post_ds41. Inputs random:
# F bf16, X bf16 (x 3), mixes U(0, 1) (post columns x 2); stats_weight random (n C)^-0.5 N(0, 1) with columns 24.. zero
# and rounded to the 9 mantissa bits the FPU's srcA multiplies (as the model loads hc_fn^T: tt/common.fpu_weight).
# Limits (F139): noise_limit at the declared formats, per output: y = X' (bf16 out), proj = stats columns 0..23 (the
# projection of the device's own bf16 X' at the call's fidelity: in0 / in1 at the mantissa bits the FPU keeps,
# reference.fpu_operands), ss = column 24 (fp32 mix squared and summed in fp32); columns 25.. exactly zero.
_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "O.1",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
    "input": {"shape": [1, 1, 1024, 2560], "dtype": "BFLOAT16", "layout": "TILE"},
    "residual": {"shape": [1, 1, 1024, 10240], "dtype": "BFLOAT16", "layout": "TILE"},
    "noise": {
        "y": {"inputs": ["bf16", "bf16", "fp32", "fp32"], "out": "bf16"},
        "proj": {"inputs": ["bf16", "fp32"], "out": "fp32"},
        "ss": {"inputs": ["bf16", "bf16", "fp32", "fp32"], "out": "fp32"},
    },
}
_DS41_STATS = {
    "stats_weight": {"shape": [1, 1, 10240, 32], "dtype": "FLOAT32", "layout": "TILE"},
    "compute_kernel_config": {"math_fidelity": "HiFi2", "math_approx_mode": False, "fp32_dest_acc_en": True},
}

CASES += [
    {
        # tt/engram_add.py: the engram add as an identity-comb mhc_post (X'_n = gate_n value + h_n) with the next hc
        # step's partial row (stats_weight); post [.., 4] / comb [.., 16] fp32 tensors of their own
        **_DS41,
        **_DS41_STATS,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-c2560-n4-bf16-stats-hifi2",
        "task": "O.1",
        "sig": "2cc390f6ee",
        "post": {"shape": [1, 1, 1024, 4], "dtype": "FLOAT32", "layout": "TILE"},
        "comb": {"shape": [1, 1, 1024, 16], "dtype": "FLOAT32", "layout": "TILE"},
        "seed": 0,
    },
    {
        # tt/residual.py:_fused (GLUE_FOLD): the residual mix with post / comb read from the packed mixes
        # [pre | post | comb] (the same tensor passed twice, post_col 4, comb_col 8), no stats
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-c2560-n4-bf16-packed-mixes",
        "sig": "6c75d98045",
        "mixes": {"shape": [1, 1, 1024, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "post_col": 4,
        "comb_col": 8,
        "seed": 1,
    },
    {
        # tt/residual.py:fused (MHC_FUSED + GLUE_FOLD): the packed mixes and the next hc step's partial row
        **_DS41,
        **_DS41_STATS,
        "id": "deepseek_v41_flash_d_p-2x2-s1024-c2560-n4-bf16-packed-mixes-stats-hifi2",
        "sig": "6f9d9f69a4",
        "mixes": {"shape": [1, 1, 1024, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "post_col": 4,
        "comb_col": 8,
        "seed": 2,
    },
]
