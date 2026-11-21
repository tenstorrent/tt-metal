# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""paged_fill_cache: write one user's K or V sequence into a paged KV cache, in place.

Specification: eval/specs/paged_fill_cache/spec.py. Design: op_design.md (this directory).

    for every position p < seq whose block b = page_table[batch_idx, p // block_size] is not -1:
        cache[b, :, p % block_size, :] = input[0, :, p, :]

The op is a copy of whole 32x32 tiles with a per-tile-row indirection read on the device: one
`ttnn.generic_op` dispatch of a Metal 2.0 ProgramSpec (reader + writer, no compute kernel), which
writes through the caller's own cache buffer and returns that tensor (rule R3).
"""

from __future__ import annotations

import ttnn

from eval import memory
from eval.specs.paged_fill_cache.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .paged_fill_cache_program_artifacts import (
    PT_RING_ENTRIES,
    RING_MAX_BYTES,
    TILE,
    create_program_artifacts,
    page_table_entry_bytes,
)

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase is the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding": ["interleaved"],
    "alignment": ["tile_aligned"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


def _shape(tensor):
    return tuple(int(d) for d in tensor.shape)


def _structural_checks(cache_tensor, input_tensor, page_table, batch_idx):
    """Malformed calls (the specification's REJECTIONS X1-X11): ValueError / TypeError."""
    # X11: batch_idx is a Python integer (not a bool, not a float).
    if isinstance(batch_idx, bool) or not isinstance(batch_idx, int):
        raise TypeError(f"paged_fill_cache: batch_idx must be an int, got {type(batch_idx).__name__} {batch_idx!r}")
    cache_shape, input_shape, pt_shape = _shape(cache_tensor), _shape(input_tensor), _shape(page_table)
    # X10: ranks.
    if len(cache_shape) != 4 or len(input_shape) != 4 or len(pt_shape) != 2:
        raise ValueError(
            f"paged_fill_cache: the cache and the input must be of rank 4 and the page table of rank 2, got "
            f"{list(cache_shape)}, {list(input_shape)}, {list(pt_shape)}"
        )
    _, heads, block_size, head_dim = cache_shape
    batch, input_heads, seq, input_dim = input_shape
    users, blocks_per_seq = pt_shape
    if batch != 1:  # X3
        raise ValueError(f"paged_fill_cache: the input's first dimension must be 1 (one user), got {batch}")
    if input_heads != heads:  # X1
        raise ValueError(f"paged_fill_cache: the input has {input_heads} heads, the cache {heads}")
    if input_dim != head_dim:  # X2
        raise ValueError(f"paged_fill_cache: the input's last dimension is {input_dim}, the cache's {head_dim}")
    if block_size % TILE != 0:  # X7
        raise ValueError(f"paged_fill_cache: the cache's block size {block_size} is not a multiple of {TILE}")
    if seq % TILE != 0:  # X8
        raise ValueError(f"paged_fill_cache: the sequence length {seq} is not a multiple of {TILE}")
    if head_dim % TILE != 0:  # X9
        raise ValueError(f"paged_fill_cache: D = {head_dim} is not a multiple of {TILE}")
    if batch_idx < 0 or batch_idx >= users:  # X4, X5
        raise ValueError(f"paged_fill_cache: batch_idx {batch_idx} is not a row of a page table of {users} rows")
    if -(-seq // block_size) > blocks_per_seq:  # X6
        raise ValueError(
            f"paged_fill_cache: a sequence of {seq} positions needs {-(-seq // block_size)} blocks of "
            f"{block_size}; a page-table row has {blocks_per_seq}"
        )


def _placement_checks(cache_tensor, input_tensor, page_table):
    """Rule R8: the cache and the page table as the callers pass them, an input of the cache's dtype."""
    for name, tensor in (("cache_tensor", cache_tensor), ("input_tensor", input_tensor), ("page_table", page_table)):
        if not ttnn.is_tensor_storage_on_device(tensor):
            raise NotImplementedError(f"paged_fill_cache: {name} must be on the device")
    if memory.memory_axes(cache_tensor) != {"memory_config": "DRAM", "sharding": "interleaved"}:
        raise NotImplementedError(f"paged_fill_cache: the cache must be DRAM interleaved, got {memory.memory_axes(cache_tensor)}")
    if cache_tensor.layout != ttnn.TILE_LAYOUT:
        raise NotImplementedError(f"paged_fill_cache: the cache must be in TILE layout, got {cache_tensor.layout}")
    if page_table.dtype != ttnn.int32 or page_table.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise NotImplementedError(
            f"paged_fill_cache: the page table must be int32 ROW_MAJOR, got {page_table.dtype} {page_table.layout}"
        )
    if memory.memory_axes(page_table) != {"memory_config": "DRAM", "sharding": "interleaved"}:
        raise NotImplementedError(
            f"paged_fill_cache: the page table must be DRAM interleaved, got {memory.memory_axes(page_table)}"
        )
    if input_tensor.dtype != cache_tensor.dtype:
        raise NotImplementedError(
            f"paged_fill_cache: the input's dtype {input_tensor.dtype} is not the cache's {cache_tensor.dtype}"
        )


def _mechanism_caps(cache_tensor, page_table):
    """Limits of the mechanisms the program uses (op_design.md, Mechanism caps)."""
    blocks, heads, block_size, head_dim = _shape(cache_tensor)
    if blocks * heads * (block_size // TILE) * (head_dim // TILE) >= 2**32:
        raise NotImplementedError("paged_fill_cache: the cache has 2^32 tiles or more (uint32 page ids)")
    if PT_RING_ENTRIES * page_table_entry_bytes(page_table) > RING_MAX_BYTES:
        raise NotImplementedError(
            f"paged_fill_cache: a page-table row of {page_table.buffer_page_size()} B does not fit the row ring"
        )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(cache_tensor, input_tensor, page_table, *, batch_idx=0):
    _structural_checks(cache_tensor, input_tensor, page_table, batch_idx)
    _placement_checks(cache_tensor, input_tensor, page_table)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        **memory.memory_axes(input_tensor),  # rule R7
    }
    inputs = (_shape(cache_tensor), _shape(input_tensor), _shape(page_table), batch_idx)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"paged_fill_cache: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"paged_fill_cache: unsupported combination (refinement candidate): {exc}")

    _mechanism_caps(cache_tensor, page_table)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def paged_fill_cache(
    cache_tensor: ttnn.Tensor,
    input_tensor: ttnn.Tensor,
    page_table: ttnn.Tensor,
    *,
    batch_idx: int = 0,
) -> ttnn.Tensor:
    """Write `input_tensor` [1, H, seq, D] into `cache_tensor` [blocks, H, block_size, D] through row
    `batch_idx` of `page_table`, in place, and return `cache_tensor` itself."""
    validate(cache_tensor, input_tensor, page_table, batch_idx=batch_idx)
    spec, run_args, tensor_indices = create_program_artifacts(cache_tensor, input_tensor, page_table, batch_idx)
    # io_tensors' last element is the "output": the caller's cache. Nothing is allocated (rule R3).
    ttnn.generic_op([input_tensor, page_table, cache_tensor], spec, run_args, tensor_indices)
    return cache_tensor
