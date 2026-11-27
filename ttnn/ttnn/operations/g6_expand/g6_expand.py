# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand: x (d0, d1, 1, W) broadcast along dim 2 to (d0, d1, H, W), as a new TILE tensor.

One device program per call (regime R1 `column_group_bcast`, op_design.md): the page axis
q = n*Wt + c is split over groups, each group a vertical strip of `row_splits` cores. The strip's
top core (the injector) reads its x pages once, copies row 0 of every page into all 32 rows by
byte copies only, and multicasts the built pages down the strip; every core of the strip writes
its share of the out tile rows straight from that one resident slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

TILE_HEIGHT = 32
TILE_WIDTH = 32
EXPAND_DIM = 2  # the only dim this contract expands
RANK = 4

# Model environment (BB-05): L1 bytes per core left to the op while the model holds its own tensors.
L1_OP_BUDGET_BYTES = 1347776

REGIME = "column_group_bcast"

# Lamp L3 (replicate realization): rows of each upper face the injector builds with RISC-V word
# stores before self-aimed NoC reads replicate them down the face. Must divide the face height (16):
# 16 is all stores, 1 is all NoC reads.
FACE_HEIGHT = 16
DEFAULT_REPLICATE_SEED_ROWS = 4


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def _out_shape(shape, sizes):
    return [int(shape[d]) if int(s) == -1 else int(s) for d, s in enumerate(sizes)]


def tag_alignment(inputs, axes):
    """Tag the OUTPUT shape: W_out misaligned wins over H_out misaligned."""
    scenario = inputs[0]
    out = _out_shape(scenario["shape"], scenario["sizes"])
    if out[-1] % TILE_WIDTH:
        return "w_non_aligned"
    if out[-2] % TILE_HEIGHT:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
# The kernels are alignment-agnostic; "h_non_aligned" is refused only because it is outside TARGET.

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_dram_interleaved(tensor):
    mem = tensor.memory_config()
    return mem.buffer_type == ttnn.BufferType.DRAM and mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def validate(input_tensor, sizes):
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_expand: input_tensor must be on a device")

    shape = list(input_tensor.shape)
    if len(shape) != RANK:
        raise ValueError(f"g6_expand: input_tensor must have rank {RANK}, got shape {shape}")
    if not isinstance(sizes, (list, tuple)) or len(sizes) != len(shape):
        raise ValueError(f"g6_expand: sizes must be a list or tuple of {len(shape)} ints, got {sizes!r}")
    for s in sizes:
        if isinstance(s, bool) or not isinstance(s, Integral):
            raise ValueError(f"g6_expand: every size must be an int, got {sizes!r}")
    for d, s in enumerate(sizes):
        if s != -1 and s <= 0:
            raise ValueError(f"g6_expand: size {s} at dim {d} must be -1 or positive")
    for d, s in enumerate(sizes):
        if shape[d] != 1 and s not in (-1, shape[d]):
            raise ValueError(f"g6_expand: size {s} at dim {d} differs from the input size {shape[d]}, which is not 1")
    for d, s in enumerate(sizes):
        if d != EXPAND_DIM and shape[d] == 1 and s not in (-1, 1):
            raise ValueError(f"g6_expand: only dim {EXPAND_DIM} may be expanded, got size {s} at dim {d}")
    if shape[EXPAND_DIM] != 1:
        raise ValueError(f"g6_expand: input dim {EXPAND_DIM} must be 1 (one row per image), got shape {shape}")
    if not _is_dram_interleaved(input_tensor):
        raise ValueError("g6_expand: input_tensor must be DRAM interleaved")

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    scenario = {"shape": shape, "sizes": list(sizes)}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Plan (BB-08): every block / core-assignment knob is set once here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpandPlan:
    regime: str
    grid: tuple
    tensor_pages: int  # Q = N * Wt
    tensor_rows_tiles: int  # Ht
    tensor_cols_tiles: int  # Wt
    strips_per_column: int  # b
    row_splits: int  # rs
    num_groups: int  # G
    group_pages_max: int
    block_pages: int
    t_block_depth: int
    num_cores: int
    l1_bytes_per_core: int
    page_bytes: int
    replicate_seed_rows: int
    core_roles: dict  # (x, y) -> "injector" | "receiver"; the role dump (BB-21)
    core_work: dict  # (x, y) -> (group_q_start, group_pages, core_r_start, core_rows_tiles)

    def knobs(self):
        return (
            self.strips_per_column,
            self.row_splits,
            self.num_groups,
            self.block_pages,
            self.t_block_depth,
            self.replicate_seed_rows,
        )


def _balanced_split(total, parts, ceil_first):
    """Contiguous balanced split of `total` units over `parts`; returns [(start, count)]."""
    base, extra = divmod(total, parts)
    counts = [base + (1 if (i < extra if ceil_first else i >= parts - extra) else 0) for i in range(parts)]
    starts = [0]
    for c in counts[:-1]:
        starts.append(starts[-1] + c)
    return list(zip(starts, counts))


def _depth_for(group_pages_max, block_pages):
    return 1 if math.ceil(group_pages_max / block_pages) == 1 else 2


def _clamp_block_pages(requested, group_pages_max, page_bytes):
    """Largest block <= requested whose CB (depth re-derived from it) fits the op's L1 budget."""
    block_pages = max(1, min(requested, group_pages_max))
    while True:
        depth = _depth_for(group_pages_max, block_pages)
        cap = L1_OP_BUDGET_BYTES // (depth * page_bytes)
        if block_pages <= cap:
            return block_pages, depth
        block_pages = max(1, cap)


def select_plan(
    input_tensor, sizes, *, strips_per_column=None, row_splits=None, block_pages=None, replicate_seed_rows=None
):
    validate(input_tensor, sizes)
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)

    out_shape = _out_shape(list(input_tensor.shape), sizes)
    page_bytes = input_tensor.buffer_aligned_page_size()
    Q = input_tensor.buffer_num_pages()
    Wt = int(input_tensor.padded_shape[-1]) // TILE_WIDTH
    Ht = math.ceil(out_shape[EXPAND_DIM] / TILE_HEIGHT)

    if strips_per_column is not None and not (1 <= strips_per_column <= gy):
        raise ValueError(f"g6_expand: strips_per_column must be in [1, {gy}], got {strips_per_column}")
    if row_splits is not None and row_splits < 1:
        raise ValueError(f"g6_expand: row_splits must be >= 1, got {row_splits}")
    if block_pages is not None and block_pages < 1:
        raise ValueError(f"g6_expand: block_pages must be >= 1, got {block_pages}")
    seed_rows = DEFAULT_REPLICATE_SEED_ROWS if replicate_seed_rows is None else replicate_seed_rows
    if seed_rows < 1 or FACE_HEIGHT % seed_rows:
        raise ValueError(f"g6_expand: replicate_seed_rows must divide {FACE_HEIGHT}, got {seed_rows}")

    def shape_for(b):
        rs = min(Ht, gy // b) if row_splits is None else min(row_splits, Ht, gy // b)
        return max(1, rs), min(Q, gx * b)

    def key(b):
        rs, G = shape_for(b)
        per_injector = math.ceil(Q / G)
        return (per_injector * math.ceil(Ht / rs), per_injector, b)

    b = strips_per_column if strips_per_column is not None else min(range(1, gy + 1), key=key)
    rs, G = shape_for(b)
    group_pages_max = math.ceil(Q / G)
    bp, depth = _clamp_block_pages(group_pages_max if block_pages is None else block_pages, group_pages_max, page_bytes)

    groups = _balanced_split(Q, G, ceil_first=True)
    splits = _balanced_split(Ht, rs, ceil_first=False)  # the injector (split 0) takes the floor share
    core_roles = {}
    core_work = {}
    for g, (q_start, q_count) in enumerate(groups):
        x = g // b
        y0 = (g % b) * rs
        for i, (r_start, r_count) in enumerate(splits):
            core = (x, y0 + i)
            core_roles[core] = "injector" if i == 0 else "receiver"
            core_work[core] = (q_start, q_count, r_start, r_count)

    return ExpandPlan(
        regime=REGIME,
        grid=(gx, gy),
        tensor_pages=Q,
        tensor_rows_tiles=Ht,
        tensor_cols_tiles=Wt,
        strips_per_column=b,
        row_splits=rs,
        num_groups=G,
        group_pages_max=group_pages_max,
        block_pages=bp,
        t_block_depth=depth,
        num_cores=G * rs,
        l1_bytes_per_core=depth * bp * page_bytes,
        page_bytes=page_bytes,
        replicate_seed_rows=seed_rows,
        core_roles=core_roles,
        core_work=core_work,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _dispatch(input_tensor, sizes, plan):
    from .g6_expand_program_descriptor import get_program_descriptor

    out_shape = _out_shape(list(input_tensor.shape), sizes)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor = get_program_descriptor(input_tensor, output_tensor, plan)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def g6_expand(input_tensor, sizes):
    validate(input_tensor, sizes)
    plan = select_plan(input_tensor, sizes)
    return _dispatch(input_tensor, sizes, plan)


def g6_expand_with_plan(input_tensor, sizes, plan):
    validate(input_tensor, sizes)
    grid = input_tensor.device().compute_with_storage_grid_size()
    Ht = math.ceil(_out_shape(list(input_tensor.shape), sizes)[EXPAND_DIM] / TILE_HEIGHT)
    if (plan.tensor_pages, plan.tensor_rows_tiles, plan.grid) != (
        input_tensor.buffer_num_pages(),
        Ht,
        (int(grid.x), int(grid.y)),
    ):
        raise ValueError("g6_expand: plan does not match the call's (pages, rows, grid)")
    return _dispatch(input_tensor, sizes, plan)
