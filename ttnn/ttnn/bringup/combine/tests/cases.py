# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.combine call a model makes (bringup-fork-tests skill). Append only; never edit
or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # MiMo-V2.6 routed experts, EP=4 on a 1x4 mesh: dispatch groups are the 4 columns, each a single chip.
        "id": "mimo_v2_6_d_p-1x4-dgs1-s5120-h4096-e256-k8",
        "model": "mimo_v2_6_d_p",
        "task": "O.1",
        "sig": "fa2baf9135",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "seq_len_per_chip": 5120,
        "emb_dim": 4096,
        "num_routed_experts": 256,  # counts / regions are [1, 256] (global expert ids)
        "num_experts_per_tok": 8,
        "experts_per_chip": 64,
        "dispatch_group_size": 1,
        "max_dispatch_buffer_token_size": 42976,
        "metadata_len": 3,
        # buffer [1, 1, 42976, 4096] BFLOAT16 TILE; metadata [1, 1, 42976, 3] INT32 ROW_MAJOR;
        # counts, regions [1, 256] UINT32 ROW_MAJOR; all DRAM interleaved
        "buffer": {"dtype": "BFLOAT16", "layout": "TILE"},
        "metadata": {"dtype": "INT32", "layout": "ROW_MAJOR"},
        "counts": {"dtype": "UINT32", "layout": "ROW_MAJOR"},
        "regions": {"dtype": "UINT32", "layout": "ROW_MAJOR"},
        "cluster_axis": 0,
        "num_links": 1,
        "topology": "Linear",
        "memory_config": "DRAM",
        "init_zeros": True,
        "use_fp8_combine": False,
        "seed": 0,
        "exact": True,
    },
]
