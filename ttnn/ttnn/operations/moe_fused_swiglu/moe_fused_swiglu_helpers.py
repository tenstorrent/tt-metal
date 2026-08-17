# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side contract and weight-placement helpers for ``moe_fused_swiglu``.

The executable operation and its program factory live in C++.  This module intentionally contains
only pure Python helpers used to validate the public API and construct the preferred DRAM weight
placement; it does not assemble kernels or allocate hidden operation inputs.
"""

from __future__ import annotations

import ttnn

from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_geometry import Blocking, TILE


# All three weights must use one common dtype.  The C++ operation independently validates the same
# set, while this tuple is used by the Python contract registry before dispatch.
WEIGHT_DTYPES = (ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16)


def worker_grid(device, core_grid=None):
    """Return the operation's ``(columns, rows)`` worker rectangle.

    By default the operation uses the complete compute-with-storage grid. An explicit ``core_grid``
    selects a smaller rectangular prefix, matching the standard C++ entry point.
    """

    grid = device.compute_with_storage_grid_size()
    max_columns, max_rows = int(grid.x), int(grid.y)
    columns, rows = max_columns, max_rows
    if core_grid is not None:
        requested_columns, requested_rows = (
            (int(core_grid.x), int(core_grid.y)) if hasattr(core_grid, "x") else (int(core_grid[0]), int(core_grid[1]))
        )
        if requested_columns < 1 or requested_rows < 1:
            raise ValueError(
                "moe_fused_swiglu: core_grid must be positive, " f"got {requested_columns}x{requested_rows}"
            )
        if requested_columns > max_columns or requested_rows > max_rows:
            raise ValueError(
                f"moe_fused_swiglu: requested grid {requested_columns}x{requested_rows} exceeds "
                f"device grid {int(grid.x)}x{int(grid.y)}"
            )
        columns, rows = requested_columns, requested_rows
    return columns, rows


def nd_shard_n_tiles(tensor):
    """Return N-axis tiles per DRAM ND shard, or zero when no contiguous run is proven."""

    try:
        memory_config = tensor.memory_config()
        shard_spec = memory_config.nd_shard_spec
        if shard_spec is None or memory_config.buffer_type != ttnn.BufferType.DRAM:
            return 0
        shard_shape = list(shard_spec.shard_shape)
    except Exception:  # pragma: no cover - permits tensor-like test doubles without these fields
        return 0
    if len(shard_shape) < 2 or int(shard_shape[-1]) % TILE != 0:
        return 0
    return int(shard_shape[-1]) // TILE


def weight_memory_configs(device, emb, hidden, core_grid=None, shard_height_tiles=1):
    """Construct the preferred ``(gate/up, down)`` DRAM ND-sharded placements.

    Each shard width is the N slice consumed by one core for one K row.  A one-tile shard height
    rotates consecutive K rows across DRAM banks.
    """

    columns, rows = worker_grid(device, core_grid)
    blocking = Blocking(columns, rows, emb, hidden, m_t_max=1)
    dram = device.dram_grid_size()
    bank_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])

    def memory_config(n_tiles):
        return ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([shard_height_tiles * TILE, n_tiles * TILE]),
                grid=bank_grid,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    return memory_config(blocking.hn_pad), memory_config(blocking.wd_ec_max)
