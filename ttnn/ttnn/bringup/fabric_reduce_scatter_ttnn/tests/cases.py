# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.fabric_reduce_scatter call a model makes (bringup-fork-tests skill). Append only;
never edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json): the input
[1, 1, G * Sb, W] bf16 TILE DRAM interleaved; the op allocates the output [1, 1, Sb, W]."""

CASES = [
    {
        # GLM-5.3-Flash common.scatter_rows (GLM_SCATTER_OP=fabric_ring, the default): the dense MLP's / shared
        # expert's per-chip partial sums over all 5120 chunk rows -> each chip's 1280-row quarter of the 4-chip sum,
        # one call over the whole mesh as a snake ring.
        "id": "glm53_flash_d_p-2x2-fullmesh-ring-s5120-w4096",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "2404ee3ae7",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "input": {"shape": [1, 1, 5120, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "cluster_axis": None,
        "topology": "Ring",
        "num_links": 2,
        "seed": 0,
        # float64 sum of the same bf16 partials; three bf16 adds per output (partials up to ~4x the result). Measured
        # (seed 0, 4 chips): pcc 0.9999970, rel L2 3.03e-3, max abs err 0.0625-0.0781, max(err - 0.008 |want|)
        # 0.021-0.023. Limits: rel 0.005, atol 0.05 + rtol 0.008; an output scaled by 1.01 fails (checked by hand).
        "pcc": 0.9999,
        "rel": 0.005,
        "atol": 0.05,
        "rtol": 0.008,
    },
    {
        # GLM-5.3-Flash KDA output projection (kda_attention: o4 -> fabric_reduce_scatter on the TP axis): per-chip
        # partials of its mesh row's 2560 rows -> each chip's 1280 rows of the row's 2-chip sum (topology default,
        # Linear: a group of 2).
        "id": "glm53_flash_d_p-2x2-axis1-s2560-w4096",
        "model": "glm53_flash_d_p",
        "task": "O.1",
        "sig": "fd313816b2",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "input": {"shape": [1, 1, 2560, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "cluster_axis": 1,
        "topology": None,
        "num_links": 2,
        "seed": 0,
        # float64 sum of the same bf16 partials; one bf16 add per output. Measured (seed 0, 4 chips): pcc 0.9999987,
        # rel L2 1.80e-3, max abs err 0.0176, every error within 0.008 |want|. Limits: rel 0.003, atol 0.01 + rtol
        # 0.008; an output scaled by 1.01 fails (checked by hand).
        "pcc": 0.9999,
        "rel": 0.003,
        "atol": 0.01,
        "rtol": 0.008,
    },
]
