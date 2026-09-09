# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor (CBs, kernels, work split).

Implements the ``grid2d_full_width`` / ``grid2d_width_chunked`` regimes of
``op_design.md``: the output tile grid ``R x C`` is blocked on BOTH axes and the
linearized block id ``row_group * num_w_chunks + w_chunk`` is what
``split_work_to_cores`` hands to the cores. ``num_w_chunks == 1`` is the
full-width parameterization; nothing else distinguishes the two regimes.

Every block knob below is a named host constant or a derived quantity with a
single source. CB page counts, loop trip counts and grid sizing are computed
FROM those knobs — never from a whole-op dimension and never as a literal.

Deviation from op_design.md (recorded here and in l1_ledger.md), one item:

  * ``block_width_tiles`` is constrained to be a DIVISOR of ``C`` instead of
    ``ceil(C / num_w_chunks_target)``, which removes ``block_width_tail_tiles``
    (every w-chunk is exactly ``block_width_tiles`` wide).
    Reason — a HARD mechanism cap the design's ragged tail violates: both CB
    endpoints refuse to wrap mid-transfer. ``llk_push_tiles``
    ``LLK_ASSERT(remaining >= num_words)`` (``llk_io_pack.h``) and
    ``cb_pop_front``'s ``ASSERT(fifo_rd_ptr <= fifo_limit)``
    (``dataflow_api.h:269``, commented "consumer always reads from contiguous
    memory, it cannot wrap") both require the CB's page count to be an exact
    multiple of every push/pop quantum used on that core. A core CAN be handed
    one full-width block and one tail block (``split_work_to_cores`` ranges are
    contiguous over a linearization that crosses w-chunk boundaries), so a
    per-core mix of two quanta is reachable and would need the capacity to be a
    multiple of ``lcm(block_width_tiles, tail)``. Making the extent a divisor of
    ``C`` costs at most a coarser chunk count and reproduces the design's whole
    worked table (``[1,1,32,16384]`` -> 8, ``[1,1,1024,1024]`` -> 16,
    ``[1,1,2048,2048]`` -> 64, ...) except ``[1,1,1,50304]`` (12 x 131 chunks
    rather than 25 x 63). The knob stays a live tunable at its coarsest correct
    value; only the tail vanishes.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB slots (semantic names; the numeric slot is just the buffer index) ---
CB_INPUT_ROWS = 0  # reader -> compute: row-major sub-block, one page per tile
CB_OUTPUT_TILES = 1  # compute -> writer: whole output tile pages

# ---------------------------------------------------------------------------
# Named block knobs — the single source of each. None is a tensor dimension.
# ---------------------------------------------------------------------------
TILE_WIDTH = 32
# Output tile-page writes to keep in flight behind one write barrier. Sets
# `write_rows_per_barrier`, which is inert (== 1) once the block is >= this wide.
WRITE_BATCH_MIN_TILES = 8
# can_use_fast_tilize requires block_width_tiles < 256 (tilize_helpers.inl:77).
FAST_TILIZE_WIDTH_CAP = 255
# low_l1=True cap on the column extent. A host constant, independent of EVERY
# tensor dimension — that independence is the low_l1 contract, structurally.
LOW_L1_WIDTH_CAP = 4
# cb_input_rows depth, in tile-rows of the block (reader/compute overlap).
INPUT_DEPTH_ROWS = 2
# cb_output_tiles depth, in WRITE BATCHES (one batch = write_rows_per_barrier
# tile-rows). Depth 2 buys the writer a full batch in flight while compute
# fills the next one, and keeps the capacity an exact multiple of the batch so
# a full batch never straddles the FIFO wrap.
OUTPUT_DEPTH_BATCHES = 2

# Legal output tile heights (power-of-two fractions of 32).
LEGAL_TILE_HEIGHTS = (1, 2, 4, 8, 16, 32)


def _largest_divisor_at_most(n: int, limit: int) -> int:
    """Coarsest divisor of `n` that is <= `limit`. Always exists (1 | n)."""
    limit = max(1, min(int(limit), int(n)))
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1


class TilizePlan:
    """The derived block plan: every knob, plus the per-core block assignment.

    Memoized (see `_PLAN_CACHE`) so a repeat call at the same shape / dtype /
    memory_config / tile / low_l1 / grid does NOT re-enter the W_FIT solve or
    `split_work_to_cores` — that derivation is exactly what the program-cache
    rule says must not be redone per call.
    """

    __slots__ = (
        "tile_h",
        "tensor_row_blocks",
        "tensor_col_tiles",
        "block_width_tiles",
        "num_w_chunks",
        "num_row_groups",
        "num_blocks_total",
        "write_rows_per_barrier",
        "input_depth_rows",
        "output_depth_batches",
        "in_page_bytes",
        "out_page_bytes",
        "block_row_bytes",
        "all_cores",
        "assignment",
        "fp32_dest_acc_en",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    # --- derived L1 footprint, in pages and bytes --------------------------
    @property
    def input_cb_pages(self) -> int:
        return self.input_depth_rows * self.block_width_tiles

    @property
    def output_cb_pages(self) -> int:
        return self.output_depth_batches * self.write_rows_per_barrier * self.block_width_tiles

    @property
    def l1_per_core_bytes(self) -> int:
        return self.input_cb_pages * self.in_page_bytes + self.output_cb_pages * self.out_page_bytes


_PLAN_CACHE: dict = {}


def plan_cache_key(input_tensor, output_tensor, *, low_l1: bool, grid) -> tuple:
    """Hashable identity of everything the plan derivation reads."""
    return (
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        tuple(output_tensor.padded_shape),
        output_tensor.dtype,
        str(output_tensor.memory_config()),
        output_tensor.buffer_page_size(),
        (output_tensor.tile.tile_shape[0], output_tensor.tile.tile_shape[1]),
        bool(low_l1),
        (grid.x, grid.y),
    )


def derive_plan(input_tensor, output_tensor, *, low_l1: bool, grid) -> TilizePlan:
    """Rule 2, in its two ordered steps: (1) fill the grid, (2) take the
    coarsest block that fits L1. Nothing here is restated from a literal."""
    key = plan_cache_key(input_tensor, output_tensor, low_l1=low_l1, grid=grid)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    num_cores = int(grid.x) * int(grid.y)

    tile_h = int(output_tensor.tile.tile_shape[0])
    shape = list(input_tensor.padded_shape)

    # Geometry: R is per-image and folds the leading dims. NEVER
    # floor(num_images * H / tile_h) -- [8,1,249,2048] is the case that differs.
    num_images = 1
    for d in shape[:-2]:
        num_images *= int(d)
    rows_per_image = math.ceil(int(shape[-2]) / tile_h)
    tensor_row_blocks = num_images * rows_per_image  # R
    tensor_col_tiles = math.ceil(int(shape[-1]) / TILE_WIDTH)  # C

    # Page bytes. tb_in is one tile's worth of ROW-MAJOR bytes; tb_out is one
    # whole output tile page, read off the buffer so block-float / tiny-tile
    # page sizes are exact rather than computed.
    in_page_bytes = tile_h * TILE_WIDTH * input_tensor.element_size()
    out_page_bytes = output_tensor.buffer_page_size()

    # --- the column-extent L1 bound, in closed form -----------------------
    # footprint = INPUT_DEPTH_ROWS*W*tb_in + OUTPUT_DEPTH_BATCHES*wrpb*W*tb_out
    # and wrpb*W <= WRITE_BATCH_MIN_TILES + W, so
    #   footprint <= W*(2*tb_in + 2*tb_out) + 2*WRITE_BATCH_MIN_TILES*tb_out
    # which inverts to W_FIT below. Budget is read from the device, never a
    # literal; no tensor dimension appears.
    budget = ttnn.get_max_worker_l1_unreserved_size()
    denom = INPUT_DEPTH_ROWS * in_page_bytes + OUTPUT_DEPTH_BATCHES * out_page_bytes
    headroom = budget - OUTPUT_DEPTH_BATCHES * WRITE_BATCH_MIN_TILES * out_page_bytes
    w_fit = max(1, min(headroom // denom, FAST_TILIZE_WIDTH_CAP))
    w_cap = min(w_fit, LOW_L1_WIDTH_CAP) if low_l1 else w_fit

    # Step 1 (occupancy) and step 2 (L1 fit) -> the smallest column split that
    # satisfies both. num_w_chunks is MINIMIZED: a column cut multiplies the
    # read transaction count, so it is spent only where it is required.
    w_chunks_for_l1 = math.ceil(tensor_col_tiles / w_cap)
    w_chunks_for_occupancy = min(tensor_col_tiles, math.ceil(num_cores / tensor_row_blocks))
    num_w_chunks_target = max(w_chunks_for_l1, w_chunks_for_occupancy)

    # Coarsest DIVISOR of C that both fits L1 and yields >= the target chunks.
    # A divisor (rather than a ceil) is what removes the ragged column tail —
    # see this module's docstring for the mechanism cap that forces it.
    width_limit = min(w_cap, tensor_col_tiles // num_w_chunks_target)
    block_width_tiles = _largest_divisor_at_most(tensor_col_tiles, width_limit)
    num_w_chunks = tensor_col_tiles // block_width_tiles  # exact, no tail

    num_row_groups = min(tensor_row_blocks, max(1, math.ceil(num_cores / num_w_chunks)))
    num_blocks_total = num_row_groups * num_w_chunks

    # Transactions-in-flight knob for the writer. Inert (1) once the block is
    # at least WRITE_BATCH_MIN_TILES wide; the whole knob on a C == 1 tensor.
    write_rows_per_barrier = max(1, math.ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))

    # --- core assignment over the linearized block id ---------------------
    # row_wise=True explicitly: a column line of cores measured 2.91x worse
    # than a row line on an interleaved DRAM->DRAM copy, and row_wise=False is
    # the default that hands you the column.
    (
        _num_cores_used,
        all_cores,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(grid, num_blocks_total, row_wise=True)

    assignment = []
    start = 0
    for group, per_core in ((core_group_1, blocks_per_core_g1), (core_group_2, blocks_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == num_blocks_total, f"tilize: block assignment covered {start} of {num_blocks_total} blocks"

    plan = TilizePlan(
        tile_h=tile_h,
        tensor_row_blocks=tensor_row_blocks,
        tensor_col_tiles=tensor_col_tiles,
        block_width_tiles=block_width_tiles,
        num_w_chunks=num_w_chunks,
        num_row_groups=num_row_groups,
        num_blocks_total=num_blocks_total,
        write_rows_per_barrier=write_rows_per_barrier,
        input_depth_rows=INPUT_DEPTH_ROWS,
        output_depth_batches=OUTPUT_DEPTH_BATCHES,
        in_page_bytes=in_page_bytes,
        out_page_bytes=out_page_bytes,
        block_row_bytes=block_width_tiles * TILE_WIDTH * input_tensor.element_size(),
        all_cores=all_cores,
        assignment=assignment,
        fp32_dest_acc_en=(input_tensor.dtype == ttnn.float32),
    )
    _PLAN_CACHE[key] = plan
    return plan


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    low_l1: bool = False,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()  # never a hardcoded core count
    plan = derive_plan(input_tensor, output_tensor, low_l1=low_l1, grid=grid)

    tile_desc = ttnn.TileDescriptor(plan.tile_h, TILE_WIDTH)

    # ========== Circular buffers ==========
    # cb_input_rows: live set is ONE tile-row of the block (the tilize helper
    # waits/pops exactly block_width_tiles pages per iteration), so
    # block_row_extent does not enter the size. Capacity = depth * live set.
    cb_input_rows = ttnn.CBDescriptor(
        total_size=plan.input_cb_pages * plan.in_page_bytes,
        core_ranges=plan.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_ROWS,
                data_format=input_tensor.dtype,
                page_size=plan.in_page_bytes,
                tile=tile_desc,
            )
        ],
    )

    # cb_output_tiles: live set is one write batch; capacity is
    # OUTPUT_DEPTH_BATCHES of them. Its data_format is where the
    # value-preserving `dtype=` cast happens, at pack time.
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=plan.output_cb_pages * plan.out_page_bytes,
        core_ranges=plan.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=plan.out_page_bytes,
                tile=tile_desc,
            )
        ],
    )

    # ========== Kernels ==========
    # The CT "plan" block is identical in all three kernels: each derives its
    # own view of a block from the same numbers, which is why there is no
    # cross-kernel handshake and no coordinator core.
    reader_ct_args = [
        CB_INPUT_ROWS,
        plan.block_width_tiles,
        plan.tile_h,
        plan.tensor_row_blocks,
        plan.num_row_groups,
        plan.num_w_chunks,
        plan.block_row_bytes,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.tensor_row_blocks,
        plan.tensor_col_tiles,
        plan.num_row_groups,
        plan.num_w_chunks,
        plan.write_rows_per_barrier,
        plan.out_page_bytes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [
        CB_INPUT_ROWS,
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.tensor_row_blocks,
        plan.num_row_groups,
        plan.num_w_chunks,
    ]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, start_block_id, num_blocks in plan.assignment:
        reader_rt_args[core.x][core.y] = [in_addr, start_block_id, num_blocks]
        writer_rt_args[core.x][core.y] = [out_addr, start_block_id, num_blocks]
        compute_rt_args[core.x][core.y] = [start_block_id, num_blocks]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=plan.fp32_dest_acc_en),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_input_rows, cb_output_tiles],
    )
