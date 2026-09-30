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
    {
        # Gemma-4 A4B routed experts, EP=4 on a 1x4 mesh: dispatch groups are the 4 columns, each a single chip.
        # The expert-output buffer is BFLOAT8_B (the routed-expert op's default output); combine unpacks it to bf16.
        "id": "gemma4_a4b_d_p-1x4-dgs1-s5120-h2816-e128-k8-bfp8",
        "model": "gemma4_a4b_d_p",
        "task": "O.1",
        "sig": "4600d93ca9",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "seq_len_per_chip": 5120,
        "emb_dim": 2816,
        "num_routed_experts": 128,  # counts / regions are [1, 128] (global expert ids)
        "num_experts_per_tok": 8,
        "experts_per_chip": 32,
        "dispatch_group_size": 1,
        "max_dispatch_buffer_token_size": 41952,
        "metadata_len": 3,
        # buffer [1, 1, 41952, 2816] BFLOAT8_B TILE; metadata [1, 1, 41952, 3] INT32 ROW_MAJOR;
        # counts, regions [1, 128] UINT32 ROW_MAJOR; all DRAM interleaved
        "buffer": {"dtype": "BFLOAT8_B", "layout": "TILE"},
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
    {
        # ERNIE-4.5 (21B-A3B) routed experts, EP=4 on a 1x4 mesh: dispatch groups are the 4 columns, each a single
        # chip. The model opens its mesh with FABRIC_1D_RING (models/demos/ernie45_d_p/tt/common.py DEVICE_PARAMS).
        # The expert-output buffer is BFLOAT8_B (unified_routed_expert_moe's output); combine unpacks it to bf16.
        "id": "ernie45_d_p-1x4-dgs1-s5120-h2560-e64-k6-bfp8",
        "model": "ernie45_d_p",
        "task": "O.1",
        "sig": "e0cf1f2a07",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_1D_RING", "l1_small_size": 24576},
        "seq_len_per_chip": 5120,
        "emb_dim": 2560,
        "num_routed_experts": 64,  # counts / regions are [1, 64] (global expert ids)
        "num_experts_per_tok": 6,
        "experts_per_chip": 16,
        "dispatch_group_size": 1,
        "max_dispatch_buffer_token_size": 31200,
        "metadata_len": 3,
        # buffer [1, 1, 31200, 2560] BFLOAT8_B TILE; metadata [1, 1, 31200, 3] INT32 ROW_MAJOR;
        # counts, regions [1, 64] UINT32 ROW_MAJOR; all DRAM interleaved
        "buffer": {"dtype": "BFLOAT8_B", "layout": "TILE"},
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
    {
        # MiMo-V2.6 routed experts on a 2x2 mesh: dispatch groups are the 2 columns of 2 chips each (cluster_axis 0,
        # fabric on, Linear); the expert-output buffer is bf16 (unified_routed_expert_moe high_precision).
        "id": "mimo_v2_6_d_p_2x2-2x2-dgs2-s2560-h4096-e256-k8",
        "model": "mimo_v2_6_d_p_2x2",
        "task": "O.1",
        "sig": "3f5a6dfceb",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "seq_len_per_chip": 2560,
        "emb_dim": 4096,
        "num_routed_experts": 256,  # counts / regions are [1, 256] (global expert ids)
        "num_experts_per_tok": 8,
        "experts_per_chip": 64,
        "dispatch_group_size": 2,
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
    {
        # Hy4 (preview) routed experts on a 2x2 mesh: dispatch groups are the 2 columns of 2 chips each (cluster_axis
        # 0, fabric on, Linear); the expert-output buffer is bf16 (unified_routed_expert_moe high_precision).
        "id": "hy4_preview_d_p-2x2-dgs2-s2560-h6144-e256-k8",
        "model": "hy4_preview_d_p",
        "task": "O.1",
        "sig": "9891342eaa",
        "mesh": [2, 2],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "seq_len_per_chip": 2560,
        "emb_dim": 6144,
        "num_routed_experts": 256,  # counts / regions are [1, 256] (global expert ids)
        "num_experts_per_tok": 8,
        "experts_per_chip": 64,
        "dispatch_group_size": 2,
        "max_dispatch_buffer_token_size": 42976,
        "metadata_len": 3,
        # buffer [1, 1, 42976, 6144] BFLOAT16 TILE; metadata [1, 1, 42976, 3] INT32 ROW_MAJOR;
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
