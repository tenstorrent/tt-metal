# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute: torch.permute(x, dims) as one device program (registry model).

Regimes, selected on the host by (layout, canonical dims), each in DRAM or L1 (interleaved):
- rows_tile_scatter: TILE bfloat16 rank 4, dims (0, 2, 1, 3);
- blocks_rm_gather: ROW_MAJOR bfloat16 rank 6, dims (0, 1, 4, 2, 5, 3);
- nhwc_tile_bfp8_transpose: TILE bfloat8_b rank 4, dims (0, 3, 1, 2).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_permute_blocks_rm_gather as blocks_rm_gather
from . import g6_permute_nhwc_tile_bfp8 as nhwc_tile_bfp8_transpose
from . import g6_permute_program_descriptor as rows_tile_scatter
from .g6_permute_blocks_rm_gather import BlocksPlan
from .g6_permute_nhwc_tile_bfp8 import NhwcPlan
from .g6_permute_program_descriptor import (
    READER_KERNEL_INDEX,
    TILE_ROWS,
    WRITER_KERNEL_INDEX,
    Plan,
    worker_grid,
)

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}


def _canonical_dims(rank, dims):
    return tuple(int(d) + rank if int(d) < 0 else int(d) for d in dims)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is (scenario,), scenario a dict with keys
#    input_shape, dims, dtype, layout, buffer.
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_buffer(inputs, axes):
    return inputs[0]["buffer"]


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_alignment(inputs, axes):
    shape = inputs[0]["input_shape"]
    H, W = int(shape[-2]), int(shape[-1])
    if W % 32 != 0:
        return "w_non_aligned"
    if H % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_dims(inputs, axes):
    """Canonical dims (negative entries wrapped); malformed dims raise ValueError before any support check."""
    scenario = inputs[0]
    rank = len(scenario["input_shape"])
    try:
        entries = [int(d) for d in scenario["dims"]]
    except TypeError as exc:
        raise ValueError(f"g6_permute: dims must be a sequence of ints, got {scenario['dims']!r}") from exc
    if len(entries) != rank:
        raise ValueError(f"g6_permute: dims {tuple(entries)} has {len(entries)} entries, the tensor has rank {rank}")
    for d in entries:
        if d < -rank or d >= rank:
            raise ValueError(f"g6_permute: dims entry {d} is outside [-{rank}, {rank - 1}]")
    canonical = _canonical_dims(rank, entries)
    if sorted(canonical) != list(range(rank)):
        raise ValueError(f"g6_permute: dims {tuple(entries)} is not a permutation of range({rank})")
    return canonical


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "buffer": tag_buffer,
    "rank": tag_rank,
    "alignment": tag_alignment,
    "dims": tag_dims,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "buffer": ["dram", "l1"],
    "rank": [4, 6],
    # rows_tile_scatter keeps W, so padding columns land in output padding columns at every alignment;
    # ROW_MAJOR pages carry no padding.
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "dims": [(0, 2, 1, 3), (0, 1, 4, 2, 5, 3), (0, 3, 1, 2)],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

# Each regime serves one (layout, dtype) per dims value: these cells have no regime and no production case.
EXCLUSIONS = [
    {"layout": ttnn.TILE_LAYOUT, "dims": (0, 1, 4, 2, 5, 3)},
    {"layout": ttnn.ROW_MAJOR_LAYOUT, "dims": (0, 2, 1, 3)},
    {"dtype": ttnn.bfloat8_b, "dims": (0, 2, 1, 3)},
    {"dtype": ttnn.bfloat8_b, "dims": (0, 1, 4, 2, 5, 3)},
    {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "dims": (0, 3, 1, 2)},
    {"layout": ttnn.ROW_MAJOR_LAYOUT, "dims": (0, 3, 1, 2)},
    # Input padding rows (W not a multiple of 32) become padding columns inside output bfloat8_b
    # groups, where a garbage value would raise the group's shared exponent.
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer_label(input_tensor):
    mem = input_tensor.memory_config()
    name = _BUFFER_NAME.get(mem.buffer_type, str(mem.buffer_type))
    if mem.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        # A sharded tensor gets its own label so SUPPORTED refuses it instead of mis-addressing it.
        return f"{name}_{str(mem.memory_layout).split('.')[-1].lower()}"
    return name


def _scenario(input_tensor, dims):
    return {
        "input_shape": tuple(int(d) for d in input_tensor.shape),
        "dims": tuple(dims) if not isinstance(dims, int) else dims,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "buffer": _buffer_label(input_tensor),
    }


def validate(input_tensor, dims):
    scenario = _scenario(input_tensor, dims)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    if input_tensor.layout == ttnn.TILE_LAYOUT and tuple(input_tensor.tile.tile_shape) != (TILE_ROWS, TILE_ROWS):
        raise ValueError(f"g6_permute: tile shape {tuple(input_tensor.tile.tile_shape)} is not 32x32")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_permute: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_permute: unsupported combination (refinement candidate): {exc}")

    return axes


# ---------------------------------------------------------------------------
# Plan report and descriptor cache
# ---------------------------------------------------------------------------

_REGIMES = {
    (ttnn.TILE_LAYOUT, (0, 2, 1, 3)): rows_tile_scatter,
    (ttnn.ROW_MAJOR_LAYOUT, blocks_rm_gather.DIMS): blocks_rm_gather,
    (ttnn.TILE_LAYOUT, nhwc_tile_bfp8_transpose.DIMS): nhwc_tile_bfp8_transpose,
}


def _select_regime(input_tensor, canonical_dims):
    """The regime module (build_plan, create_program_descriptor, knob_key) of a validated call."""
    return _REGIMES[(input_tensor.layout, tuple(canonical_dims))]


# One (plan, descriptor) per cache key; a hit only patches the two buffer addresses.
_PROGRAM_CACHE: dict = {}


def _cache_key(input_tensor, canonical_dims):
    mem = input_tensor.memory_config()
    grid = worker_grid(input_tensor.device())
    regime = _select_regime(input_tensor, canonical_dims)
    return (
        regime.REGIME,
        regime.knob_key(),
        input_tensor.buffer_page_size(),
        input_tensor.buffer_aligned_page_size(),
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        tuple(canonical_dims),
        input_tensor.dtype,
        input_tensor.layout,
        mem.buffer_type,
        mem.memory_layout,
        tuple(input_tensor.tile.tile_shape),
        (grid.x, grid.y),
    )


def g6_permute_plan(input_tensor: ttnn.Tensor, dims) -> Plan | BlocksPlan | NhwcPlan:
    """Report the plan a call selects (regime, active cores, block extent, blocks per core group,
    per-core roles) without dispatching."""
    axes = validate(input_tensor, dims)
    entry = _PROGRAM_CACHE.get(_cache_key(input_tensor, axes["dims"]))
    if entry is not None:
        return entry[0]
    return _select_regime(input_tensor, axes["dims"]).build_plan(input_tensor, axes["dims"])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_permute(input_tensor: ttnn.Tensor, dims) -> ttnn.Tensor:
    axes = validate(input_tensor, dims)
    canonical_dims = axes["dims"]

    out_shape = [int(input_tensor.shape[d]) for d in canonical_dims]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, canonical_dims)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        regime = _select_regime(input_tensor, canonical_dims)
        plan = regime.build_plan(input_tensor, canonical_dims)
        descriptor = regime.create_program_descriptor(plan, input_tensor, output_tensor)
        _PROGRAM_CACHE[key] = (plan, descriptor)
    else:
        _, descriptor = entry
        descriptor.kernels[READER_KERNEL_INDEX].common_runtime_args = [input_tensor.buffer_address()]
        descriptor.kernels[WRITER_KERNEL_INDEX].common_runtime_args = [output_tensor.buffer_address()]

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
