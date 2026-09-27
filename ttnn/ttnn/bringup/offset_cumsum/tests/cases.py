# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.offset_cumsum call a model makes (bringup-fork-tests skill). Append only; never
edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

CASES = [
    {
        # MiMo-V2.6 routed experts on a 1x4 mesh: cluster_axis 0 has one device, so each device is its own group.
        "id": "mimo_v2_6_d_p-1x4-axis0-e256-epc64",
        "model": "mimo_v2_6_d_p",
        "task": "O.1",
        "sig": "70b8c49b0a",
        "mesh": [1, 4],
        "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
        "hist_shape": [256],  # per device, UINT32 ROW_MAJOR DRAM interleaved (masked_bincount output)
        "hist": {"dtype": "UINT32", "layout": "ROW_MAJOR"},
        # Histogram values: a device's 5120 tokens x top-8 over its 64 local experts (the others 0), drawn at
        # random below; the op is exact for any count, so the range only has to stay realistic.
        "max_count": 5120,
        "local_experts_only": True,
        "cluster_axis": 0,
        "num_links": 1,
        "experts_per_chip": 64,
        "memory_config": "DRAM",
        "seed": 0,
        "exact": True,
    },
]
