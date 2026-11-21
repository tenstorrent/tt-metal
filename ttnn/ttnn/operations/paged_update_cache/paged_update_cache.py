# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""paged_update_cache: one decode step's head rows written into a paged, tile-layout KV cache, in place.

Specification: eval/specs/paged_update_cache/spec.py. Design: op_design.md (regime `direct_row_write`).

    for each user u with p = update_idxs_tensor[u] != -1, each head h < heads, each d < D:
        cache[page_table[u, p // block_size], h, p % block_size, d] = input[0, u, h, d]

The call is one `ttnn.generic_op` dispatch of one data-movement kernel on the cores holding the input's
shards; the cache is the caller's tensor, written in place and returned (rule R3).
"""

from __future__ import annotations

import ttnn

from eval import memory
from eval.specs.paged_update_cache.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .paged_update_cache_program_artifacts import (
    TILE_DIM,
    create_program_artifacts,
    index_scratch_bytes,
    l1_index_budget,
)

OP_NAME = "paged_update_cache"

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase, equal to the specification's PHASE0.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_config": ["L1"],
    "sharding": ["height"],
    "users": [1],
    "alignment": ["h_non_aligned"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shape(tensor):
    return tuple(int(d) for d in tensor.shape)


def _is_dram_interleaved(tensor):
    return memory.memory_axes(tensor) == {"memory_config": "DRAM", "sharding": "interleaved"}


def _check_malformed(cache_shape, input_shape, idx_shape, table_shape):
    """REJECTIONS X1-X8, X10, X11: malformed calls, refused with ValueError."""
    if len(cache_shape) != 4 or len(input_shape) != 4:
        raise ValueError(
            f"{OP_NAME}: the cache and the input must be of rank 4 (got {list(cache_shape)} and {list(input_shape)})"
        )
    blocks, heads, block_size, D = cache_shape
    _, users, rows, input_D = input_shape
    if input_shape[0] != 1:
        raise ValueError(f"{OP_NAME}: the input's first dimension must be 1, one token a user (got {input_shape[0]})")
    if input_D != D:
        raise ValueError(f"{OP_NAME}: the input's last dimension {input_D} is not the cache's {D}")
    if D % TILE_DIM != 0:
        raise ValueError(f"{OP_NAME}: D = {D} must be a multiple of {TILE_DIM}")
    if block_size % TILE_DIM != 0:
        raise ValueError(f"{OP_NAME}: the cache's block size {block_size} must be a multiple of {TILE_DIM}")
    if rows < heads:
        raise ValueError(f"{OP_NAME}: the input has {rows} rows per user, fewer than the cache's {heads} heads")
    if rows > TILE_DIM:
        raise ValueError(f"{OP_NAME}: the input has {rows} rows per user; at most {TILE_DIM} (one tile row)")
    if len(idx_shape) != 1 or idx_shape[0] != users:
        raise ValueError(f"{OP_NAME}: update_idxs_tensor must be [{users}], one position per user (got {list(idx_shape)})")
    if len(table_shape) != 2 or table_shape[0] != users:
        raise ValueError(f"{OP_NAME}: page_table must be [{users}, blocks_per_seq] (got {list(table_shape)})")
    if table_shape[1] > blocks:
        raise ValueError(
            f"{OP_NAME}: a page-table row has {table_shape[1]} entries, more than the cache's {blocks} blocks"
        )


def _check_placement(cache_tensor, input_tensor, update_idxs_tensor, page_table):
    """Rule R8: the one form of the cache and the index tensors this op takes."""
    if cache_tensor.layout != ttnn.TILE_LAYOUT or not _is_dram_interleaved(cache_tensor):
        raise NotImplementedError(f"{OP_NAME}: the cache must be TILE_LAYOUT, interleaved in DRAM")
    for name, t in (("update_idxs_tensor", update_idxs_tensor), ("page_table", page_table)):
        if t.dtype != ttnn.int32 or t.layout != ttnn.ROW_MAJOR_LAYOUT or not _is_dram_interleaved(t):
            raise NotImplementedError(f"{OP_NAME}: {name} must be int32, ROW_MAJOR_LAYOUT, interleaved in DRAM")
    if input_tensor.dtype != cache_tensor.dtype:
        raise NotImplementedError(
            f"{OP_NAME}: the input's dtype {input_tensor.dtype} is not the cache's {cache_tensor.dtype}"
        )


def _check_tile_geometry(cache_tensor, input_tensor):
    """Mechanism cap: the segment offsets assume 32x32 tiles of four 16x16 faces."""
    for name, t in (("cache", cache_tensor), ("input", input_tensor)):
        tile = t.tile
        if list(tile.tile_shape) != [TILE_DIM, TILE_DIM] or list(tile.face_shape) != [16, 16]:
            raise NotImplementedError(
                f"{OP_NAME}: the {name} must use 32x32 tiles of 16x16 faces "
                f"(got {list(tile.tile_shape)} / {list(tile.face_shape)})"
            )


def validate(cache_tensor, input_tensor, *, update_idxs_tensor, page_table):
    shapes = tuple(_shape(t) for t in (cache_tensor, input_tensor, update_idxs_tensor, page_table))
    _check_malformed(*shapes)
    _check_placement(cache_tensor, input_tensor, update_idxs_tensor, page_table)

    # The cell's axes: the input's dtype, layout and placement (R7), then every tagger in order.
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout, **memory.memory_axes(input_tensor)}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(shapes, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")

    _check_tile_geometry(cache_tensor, input_tensor)
    need, budget = index_scratch_bytes(input_tensor, update_idxs_tensor, page_table), l1_index_budget()
    if need > budget:
        raise NotImplementedError(
            f"{OP_NAME}: the index scratch needs {need} B per core, more than the {budget} B budget "
            "(ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES); the page-table row is too long "
            "for the direct_row_write regime"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def paged_update_cache(
    cache_tensor: ttnn.Tensor,
    input_tensor: ttnn.Tensor,
    *,
    update_idxs_tensor: ttnn.Tensor,
    page_table: ttnn.Tensor,
) -> ttnn.Tensor:
    """Write each user's head rows into the paged cache at its position, in place; return `cache_tensor`."""
    validate(cache_tensor, input_tensor, update_idxs_tensor=update_idxs_tensor, page_table=page_table)
    spec, run_args, tensor_indices = create_program_artifacts(
        cache_tensor, input_tensor, update_idxs_tensor, page_table
    )
    # The cache is the last io tensor: the program's "output" is the caller's own buffer, nothing is allocated.
    ttnn.generic_op([input_tensor, update_idxs_tensor, page_table, cache_tensor], spec, run_args, tensor_indices)
    return cache_tensor
