# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.mhc_pre call a model makes (bringup-fork-tests skill). Append only; never edit
or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w0",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "4278aef645",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.051429904997348785, 0.046146079897880554, 0.17659668624401093],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 0,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w1",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "412af7ea2f",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.061402466148138046, 0.04462148994207382, 0.08887463808059692],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 1,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w2",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "c69667d8a4",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.07067592442035675, 0.028531519696116447, 0.09720445424318314],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 2,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w3",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "e2ec8b2136",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.07561369240283966, 0.05657198280096054, 0.1186501681804657],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 3,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w4",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "05e486525c",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.07759831100702286, 0.04633378982543945, 0.15449853241443634],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 4,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w5",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "73dc6905cb",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.08108066767454147, 0.12104804813861847, 0.15734434127807617],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 5,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w6",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "a6019c38e9",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.08264460414648056, 0.08581192791461945, 0.14026999473571777],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 6,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w7",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "3f5ad27553",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.1002449169754982, 0.05559425428509712, 0.12795057892799377],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 7,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w8",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "fd47e88a7e",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.10678376257419586, 0.07812844961881638, 0.08156060427427292],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 8,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
    {
        # GLM-5.3-Flash mHC pre (attn / ffn hc + collapse) on a 5120-token chunk, split layout: each chip its 1280
        # rows of the 4 streams packed along the last dim (n*C = 4 * 4096); W replicated. One layer's scales.
        "id": "glm53_flash_d_p-2x2-s1280-nc16384-bf16-w9",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "0d6d3e6b41",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        # input [1, 1, 1280, 16384] bf16 TILE DRAM interleaved (per device)
        "input": {"shape": [1, 1, 1280, 16384], "dtype": "BFLOAT16", "layout": "TILE"},
        # proj_weight [16384, 24] bf16 TILE DRAM interleaved (replicated), proj_bias [1, 24] fp32 TILE
        "proj_weight": {"shape": [16384, 24], "dtype": "BFLOAT16", "layout": "TILE"},
        "proj_bias": {"shape": [1, 24], "dtype": "FLOAT32", "layout": "TILE"},
        "scale": [0.10733209550380707, 0.08431563526391983, 0.171678826212883],
        "sinkhorn_iters": 20,
        "eps": 1e-06,
        "norm_eps": 1e-05,
        "seed": 9,
        # measured (seeds 0-9, 4 devices): y rel L2 0.00167 (bf16 output rounding), post <= 2.2e-5, comb <= 3.9e-5,
        # pcc >= 0.9999986; a 1.01 scale of any output (rel 0.01) fails these limits
        "pcc": 0.9999,
        "max_rel": {"y": 0.004, "post": 0.0002, "comb": 0.0005},
    },
]


# The Xing4.0 entries of the fork (ttnn.bringup.mhc_pre_xing / mhc_pre_xing_pack) keep their own case list and test
# (xing_cases.py, test_mhc_pre_xing.py: "op" below); they are listed here too so the fork-call checker
# (models/demos/common/bringup/testing/fork_cases.py reads only cases.py) sees their sigs. test_mhc_pre_ttnn.py runs
# only the mhc_pre cases (no "op" key).
def _xing_cases():
    import runpy
    from pathlib import Path

    xs = runpy.run_path(str(Path(__file__).resolve().parent / "xing_cases.py"))["CASES"]
    return [{**c, "op": "mhc_pre_xing_pack" if c["mode"] == "pack" else "mhc_pre_xing"} for c in xs if "sig" in c]


CASES += _xing_cases()
