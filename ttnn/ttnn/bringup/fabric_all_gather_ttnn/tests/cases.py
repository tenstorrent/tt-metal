# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.fabric_all_gather call a model makes (bringup-fork-tests skill). Append only;
never edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json): every input
and output DRAM interleaved, a preallocated output (output_tensor)."""

_GLM = {
    "model": "glm53_flash_d_p",
    "task": "O.1",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
    "num_links": 2,
    "seed": 0,
    "exact": True,
}

CASES = [
    {
        # GLM-5.3-Flash common.gather_axis(.., 0): a mesh row's half of a 5120-token chunk -> all rows (the second step
        # of gather_rows: the hidden state before the dense MLP / shared expert).
        **_GLM,
        "id": "glm53_flash_d_p-2x2-axis0-dim2-tile-s2560-w4096",
        "sig": "5cc620ed94",
        "input": {"shape": [1, 1, 2560, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 5120, 4096]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        # GLM-5.3-Flash common.gather_axis(.., 1): this chip's 1280 rows of the q_a latent (1536) -> its mesh row's half.
        **_GLM,
        "id": "glm53_flash_d_p-2x2-axis1-dim2-tile-s1280-w1536",
        "sig": "97b27a5b33",
        "input": {"shape": [1, 1, 1280, 1536], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 2560, 1536]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        # GLM-5.3-Flash common.gather_axis(.., 1): this chip's 1280 hidden rows (4096) -> its mesh row's half (the first
        # step of gather_rows, and the KDA / MoE input gathers).
        **_GLM,
        "id": "glm53_flash_d_p-2x2-axis1-dim2-tile-s1280-w4096",
        "sig": "a1eaf79ca8",
        "input": {"shape": [1, 1, 1280, 4096], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 2560, 4096]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        # GLM-5.3-Flash MLA striped latent cache (mla_attention._kv): each chip's block-cyclic stripe of the
        # [S, 512] latent cache (20480 rows = an 81920-token worst case over 4 chips), gathered over the whole mesh
        # (snake ring) into the shared scratch at each chip's fixed slot; only the populated prefix
        # (gathered_dim_size 56320 = the last chunk's end, 14080 rows per chip) moves, the rest of the scratch is
        # left as it was.
        **_GLM,
        "id": "glm53_flash_d_p-2x2-fullmesh-dim2-rm-s20480-w512-prefix56320",
        "sig": "c2040b37ca",
        "input": {"shape": [1, 1, 20480, 512], "dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "output": {"shape": [1, 1, 81920, 512]},
        "dim": 2,
        "cluster_axis": None,
        "gathered_dim_size": 56320,
    },
]


# deepseek_v41_flash_d_p O.1 (scaled 2x2 run, ladder rung last: chunk 53248-55296, 1024 rows a chip): every gather of
# tt/ccl.py:all_gather under CCL_IMPL fabric_bf16 (the norms' stats partials, the mHC partial rows on dim 1, the hidden
# columns on dim 3, the indexer's uint32 top-k rows, the experts' q(x) rows over axis 0, ...). The model passes the
# output preallocated (allocate_tensor_on_device), its own pair of L1_SMALL global semaphores (ccl._sems: zero, the
# whole worker grid) and num_links None (every usable link of the axis); its mesh runs a 10880 B fabric packet payload
# (hooks.device_params: the weighted combine's one-row packets). Inputs random in the captured dtype (uint32: random
# 31-bit words); compared bit for bit.
_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "O.1",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576, "fabric_payload_bytes": 10880},
    "num_links": None,
    "semaphores": True,
    "seed": 0,
    "exact": True,
}

CASES += [
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim3-float32-tile-s1024-w64",
        "sig": "0350cd2422",
        "input": {"shape": [1, 1, 1024, 64], "dtype": "FLOAT32", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 128]},
        "dim": 3,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-bfloat16-tile-s512-w1792",
        "sig": "2f761f8778",
        "input": {"shape": [1, 1, 512, 1792], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 1792]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-uint32-rm-s512-w512",
        "sig": "3ed44796fc",
        "input": {"shape": [1, 1, 512, 512], "dtype": "UINT32", "layout": "ROW_MAJOR"},
        "output": {"shape": [1, 1, 1024, 512]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-float32-tile-s512-w32",
        "sig": "43f469cea2",
        "input": {"shape": [1, 1, 512, 32], "dtype": "FLOAT32", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 32]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-rm-s1024-w5120",
        "sig": "44b17650b3",
        "input": {"shape": [1, 1, 1024, 5120], "dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "output": {"shape": [1, 1, 2048, 5120]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim3-bfloat16-tile-s1024-w2560",
        "sig": "50b80cde70",
        "input": {"shape": [1, 1, 1024, 2560], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 5120]},
        "dim": 3,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim1-float32-tile-s1024-w32",
        "sig": "5215b10576",
        "input": {"shape": [1, 1, 1024, 32], "dtype": "FLOAT32", "layout": "TILE"},
        "output": {"shape": [1, 2, 1024, 32]},
        "dim": 1,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s256-w512",
        "sig": "6d75743894",
        "input": {"shape": [1, 1, 256, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 512, 512]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s512-w512",
        "sig": "743a405dcd",
        "input": {"shape": [1, 1, 512, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 512]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s1024-w128",
        "sig": "a8b1f5026d",
        "input": {"shape": [1, 1, 1024, 128], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 2048, 128]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-bfloat16-tile-s512-w1024",
        "sig": "ad4aac6edc",
        "input": {"shape": [1, 1, 512, 1024], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 1024]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-bfloat16-tile-s128-w512",
        "sig": "bc9fc305e3",
        "input": {"shape": [1, 1, 128, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 256, 512]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis1-dim2-bfloat16-tile-s512-w512",
        "sig": "d5f476e647",
        "input": {"shape": [1, 1, 512, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 512]},
        "dim": 2,
        "cluster_axis": 1,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s1024-w512",
        "sig": "e109e2154a",
        "input": {"shape": [1, 1, 1024, 512], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 2048, 512]},
        "dim": 2,
        "cluster_axis": 0,
    },
    {
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-axis0-dim2-bfloat16-tile-s512-w128",
        "sig": "ffe3863107",
        "input": {"shape": [1, 1, 512, 128], "dtype": "BFLOAT16", "layout": "TILE"},
        "output": {"shape": [1, 1, 1024, 128]},
        "dim": 2,
        "cluster_axis": 0,
    },
]
