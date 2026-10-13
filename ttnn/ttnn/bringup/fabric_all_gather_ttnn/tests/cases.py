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
