# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_untilize program descriptor: regime `row_gather`.

The output rows are split into contiguous balanced ranges over one core rectangle. On each core the
reader gathers the 64-B face-row pieces of its rows from the TILE input straight into row-major
positions of `cb_row_block`; the writer stores each row as one ROW_MAJOR page. No compute kernel.

Every block knob (core limit, block_tile_cols, buffer depth) is set here once and reaches the kernels
as a compile-time arg; the kernels derive every loop bound and per-core range from those args.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
READER_KERNEL = KERNEL_DIR / "g6_untilize_reader.cpp"
WRITER_KERNEL = KERNEL_DIR / "g6_untilize_writer.cpp"

# Block knobs. None = no cap beyond the work units and the grid.
CORE_LIMIT = None
ROW_BLOCK_DEPTH = 2
# Start each core's tile-column walk at core_idx mod block_tile_cols (gather CT switch).
BANK_STAGGER = True
# Data-movement RISCs issuing the gather: 1 = reader only (NoC0); 2 = reader on NoC0 and writer on NoC1,
# split by tile-column walk step.
GATHER_RISCS = 2
# Free L1 per core the model leaves to the op (BB-05).
L1_OP_BUDGET_BYTES = 1347776

CB_ROW_BLOCK = 0

REGIME_ROW_GATHER = "row_gather"


def select_regime(input_tensor) -> str:
    return REGIME_ROW_GATHER


def choose_core_rectangle(num_units, grid, core_limit=None):
    """Largest w x h rectangle with w*h <= min(units, grid, limit); ties go to w >= h, then squarest."""
    cap = min(num_units, grid.x * grid.y)
    if core_limit is not None:
        cap = min(cap, core_limit)
    cap = max(cap, 1)
    best = None
    for h in range(1, grid.y + 1):
        w = min(grid.x, cap // h)
        if w < 1:
            break
        key = (w * h, w >= h, -abs(w - h))
        if best is None or key > best[0]:
            best = (key, w, h)
    return best[1], best[2]


def derive_block_tile_cols(tensor_tile_cols, tile_bytes, depth, budget):
    """Largest divisor of tensor_tile_cols whose CB at full depth fits the L1 budget.

    A divisor keeps the push quantum identical for every col block, so the CB ring never wraps
    mid-block.
    """
    for cols in range(tensor_tile_cols, 0, -1):
        if tensor_tile_cols % cols == 0 and depth * cols * tile_bytes <= budget:
            return cols
    raise ValueError(f"g6_untilize: one tile column at depth {depth} exceeds the {budget}-B L1 budget")


def _plan(input_tensor):
    shape = tuple(input_tensor.shape)
    batch = 1
    for d in shape[:-2]:
        batch *= d
    rows_per_image, row_elems = shape[-2], shape[-1]
    tile_h, tile_w = input_tensor.tile.tile_shape
    tile_bytes = input_tensor.tile.get_tile_size(input_tensor.dtype)

    tensor_rows = batch * rows_per_image
    tile_rows_per_image = -(-rows_per_image // tile_h)
    tensor_tile_cols = -(-row_elems // tile_w)

    grid = input_tensor.device().compute_with_storage_grid_size()
    rect_w, rect_h = choose_core_rectangle(tensor_rows, grid, CORE_LIMIT)
    num_cores = rect_w * rect_h

    block_tile_cols = derive_block_tile_cols(tensor_tile_cols, tile_bytes, ROW_BLOCK_DEPTH, L1_OP_BUDGET_BYTES)
    num_col_blocks = tensor_tile_cols // block_tile_cols
    max_core_rows = -(-tensor_rows // num_cores)
    max_blocks_per_core = -(-max_core_rows // tile_h) * num_col_blocks
    buffer_depth = min(ROW_BLOCK_DEPTH, max_blocks_per_core)

    return {
        "regime": select_regime(input_tensor),
        "grid": (grid.x, grid.y),
        "rect_w": rect_w,
        "rect_h": rect_h,
        "num_cores": num_cores,
        "tensor_rows": tensor_rows,
        "rows_per_image": rows_per_image,
        "row_elems": row_elems,
        "tile_rows_per_image": tile_rows_per_image,
        "tensor_tile_cols": tensor_tile_cols,
        "block_tile_cols": block_tile_cols,
        "buffer_depth": buffer_depth,
        "tile_bytes": tile_bytes,
    }


def _core_rows(core_idx, tensor_rows, num_cores):
    """Contiguous balanced row range; the kernels evaluate the same formula."""
    base, extra = divmod(tensor_rows, num_cores)
    return core_idx * base + min(core_idx, extra), base + (1 if core_idx < extra else 0)


def _core_ranges(plan):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(plan["rect_w"] - 1, plan["rect_h"] - 1))]
    )


def describe_plan(input_tensor) -> dict:
    """Report the plan a call on this input selects. Dispatches nothing."""
    plan = _plan(input_tensor)
    roles = {}
    for y in range(plan["rect_h"]):
        for x in range(plan["rect_w"]):
            start, rows = _core_rows(y * plan["rect_w"] + x, plan["tensor_rows"], plan["num_cores"])
            roles[(x, y)] = {
                "reader": "gather_row_block",
                "writer": "gather_row_block+store_row_block" if GATHER_RISCS == 2 else "store_row_block",
                "row_start": start,
                "rows": rows,
            }
    return {
        "regime": plan["regime"],
        "core_ranges": _core_ranges(plan),
        "num_cores": plan["num_cores"],
        "block_tile_cols": plan["block_tile_cols"],
        "buffer_depth": plan["buffer_depth"],
        "roles": roles,
    }


def _build_program_descriptor(input_tensor, output_tensor, plan) -> ttnn.ProgramDescriptor:
    core_ranges = _core_ranges(plan)
    tile_bytes = plan["tile_bytes"]

    cb_row_block = ttnn.CBDescriptor(
        total_size=plan["block_tile_cols"] * plan["buffer_depth"] * tile_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_ROW_BLOCK, data_format=input_tensor.dtype, page_size=tile_bytes)
        ],
    )

    # Shared prefix: g6_untilize::GatherArgs reads indices 0..10 in both kernels.
    gather_ct_args = [
        CB_ROW_BLOCK,
        plan["tensor_rows"],
        plan["num_cores"],
        plan["rect_w"],
        plan["rows_per_image"],
        plan["row_elems"],
        plan["tile_rows_per_image"],
        plan["tensor_tile_cols"],
        plan["block_tile_cols"],
        int(BANK_STAGGER),
        GATHER_RISCS,
    ]
    input_accessor_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    reader_ct_args = gather_ct_args + list(input_accessor_args)
    writer_ct_args = (
        gather_ct_args
        + list(input_accessor_args)
        + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    )

    reader = ttnn.KernelDescriptor(
        kernel_source=str(READER_KERNEL),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(WRITER_KERNEL),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        common_runtime_args=[output_tensor.buffer_address(), input_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb_row_block])


READER_IDX, WRITER_IDX = 0, 1

# Built once per program-shaping key; a hit only rewrites the two buffer addresses (BB-07).
_DESCRIPTOR_CACHE: dict = {}


def _device_key(device):
    try:
        return device.id()
    except (AttributeError, TypeError):
        return id(device)


def _knob_key():
    """Every module knob that shapes the program. Read at call time, so a tuned knob reaches the cache key."""
    return (CORE_LIMIT, ROW_BLOCK_DEPTH, BANK_STAGGER, GATHER_RISCS, L1_OP_BUDGET_BYTES)


def get_program_descriptor(input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    in_mc = input_tensor.memory_config()
    key = (
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        in_mc.memory_layout,
        in_mc.buffer_type,
        _device_key(device),
        grid.x,
        grid.y,
        *_knob_key(),
    )
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        descriptor = _build_program_descriptor(input_tensor, output_tensor, _plan(input_tensor))
        _DESCRIPTOR_CACHE[key] = descriptor
        return descriptor
    # Elements of descriptor.kernels alias the stored C++ descriptors, so these writes persist.
    kernels = descriptor.kernels
    kernels[READER_IDX].common_runtime_args[0] = input_tensor.buffer_address()
    kernels[WRITER_IDX].common_runtime_args[0] = output_tensor.buffer_address()
    kernels[WRITER_IDX].common_runtime_args[1] = input_tensor.buffer_address()
    return descriptor
