# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.dispatch call a model makes (bringup-fork-tests skill). Append only; never edit
or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # MiMo-V2.6 routed experts, EP=4 on a 1x4 mesh: dispatch groups are the 4 columns, each a single chip.
        "id": "mimo_v2_6_d_p-1x4-dgs1-s5120-h4096-e256-k8",
        "model": "mimo_v2_6_d_p",
        "task": "O.1",
        "sig": "33e8891784",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "seq_len_per_chip": 5120,
        "emb_dim": 4096,
        "num_routed_experts": 256,
        "num_experts_per_tok": 8,
        "experts_per_chip": 64,
        "dispatch_group_size": 1,
        "metadata_len": 3,
        "max_dispatch_buffer_token_size": 42976,
        "cluster_axis": 0,
        "num_links": 1,
        "topology": "Linear",
        "fp8_output": False,
        "num_workers_per_sender": 2,
        "subdevice_id": None,
        # input [1, S, H], indices [1, S, K], offsets [1, E], table [1, E + 1]; all DRAM interleaved
        "input": {"dtype": "BFLOAT16", "layout": "TILE"},
        "indices": {"dtype": "UINT16", "layout": "ROW_MAJOR"},
        "offsets": {"dtype": "UINT32", "layout": "ROW_MAJOR"},
        "table": {"dtype": "INT32", "layout": "ROW_MAJOR"},
        "seed": 0,
        "exact": True,
    },
]
