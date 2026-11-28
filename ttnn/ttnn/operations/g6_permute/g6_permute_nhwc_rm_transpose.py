# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute, regime nhwc_rm_transpose: plan (unit split, unit width) and ProgramDescriptor.

ROW_MAJOR x [N, H, W, C], dims (0, 3, 1, 2): out[n, c, h, w] = x[n, h, w, c].
Input page (n, h, w) holds the C values of one pixel; output page (n, c, h) holds the W values of one
channel row. Each row (n, h) of W input pages is cut into units of block_w pages (the last unit of a
row holds the tail_w left over). Unit (n, h, wb) is the input pages [(n * H + h) * W + wb * block_w,
+ width) and writes the C partial output rows (n, c, h), values [wb * block_w, + width), at byte
offset wb * block_w * 2.

The transpose runs on the Tensix engine. The reader lands page k of a unit in slot k of
slot_bytes = ct * 64 B (ct = ceil(C / 32)), so every 32 slots are a 32-row row-major block of ct
tiles: w-tile wt of the unit. Compute tilizes the unit's kw w-tiles, transposes each tile and
pack-untilizes them, emitting per c-tile 32 rows c of kw * 32 values w; the writer sends each row
c < C as its partial output row. Slot rows past a unit's width and slot bytes past C values only
reach output positions the writer never sends.

Every CB is UInt16: the unpacker and packer then move the 16 bits opaquely. Float16_b through the
same path flushes denormals, drops NaN payloads and turns -0 into +0 (all 65536 bfloat16 patterns
measured on Blackhole), which breaks the bit-exact contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import ttnn

from .g6_permute_blocks_rm_gather import CB_INPUT_SLOTS, CB_OUTPUT_ROWS, dm_config, read_plan, reader_descriptor
from .g6_permute_program_descriptor import (
    KERNEL_DIR,
    L1_OP_BUDGET_BYTES,
    READER_KERNEL_INDEX,
    TILE_ROWS,
    WRITER_KERNEL_INDEX,
    _div_up,
    out_l1_bytes_per_core,
    split_units,
    worker_grid,
)

REGIME = "nhwc_rm_transpose"
DIMS = (0, 3, 1, 2)

COMPUTE_KERNEL_INDEX = 2
CB_TILES = 2

TILE_BYTES = TILE_ROWS * TILE_ROWS * 2
# The input slot of one page: one row of a ct-tile tilize block.
SLOT_ROW_BYTES = TILE_ROWS * 2
# A unit's w-tiles share one 16-bit half-sync DEST batch.
MAX_UNIT_TILES = 8

# Buffer-depth knobs: cb_input_slots in units, cb_output_rows in c-tile pages.
INPUT_DEPTH = 2
ROWS_DEPTH = 2
# Unit width in input pages: a multiple of 8 values (the 16 B NoC write granule of a partial row) of at
# most MAX_UNIT_TILES w-tiles, or >= W for whole rows. None lets _choose_block_w pick.
BLOCK_W = None
BLOCK_W_CHOICES = (8, 16, 32, 64, 128, 256)
# Per-core costs (cycles) of _choose_block_w, measured on p44: one page read issued, one partial-row
# write issued, one tile through tilize + transpose + untilize, one unit's fixed reader cost.
READ_CYCLES = 37
WRITE_CYCLES = 42
TILE_CYCLES = 330
UNIT_CYCLES = 200
# The shared reader's DRAM-input knobs (blocks_rm_gather measured them on 4 B pages).
READ_ALT_NOC_SHARE = Fraction(1, 2)
BANK_MAJOR = True
BANK_RUN_ROUNDS = 8
BANK_STAGGER = False


@dataclass
class NhwcRmPlan:
    regime: str
    num_cores: int
    tensor_units: int
    units_per_core_g1: int
    units_per_core_g2: int
    block_w: int
    tail_w: int
    units_per_row: int
    kw: int
    kw_tail: int
    ct: int
    input_depth: int
    rows_depth: int
    shared_reader: bool
    core_roles: dict
    shape: tuple = ()
    in_page_bytes: int = 0
    slot_bytes: int = 0
    out_page_bytes: int = 0
    out_l1_bytes_per_core: int = 0
    l1_bytes_per_core: int = 0
    read_alt_share: Fraction = Fraction(0)
    num_banks: int = 1
    bank_stagger: bool = False
    bank_major: bool = False
    chunk_pages: int = 0
    all_cores: object = None
    assignment: list = field(default_factory=list)  # [(CoreCoord, start_unit, num_units)]


def knob_key():
    """Host knobs that reach this regime's CT args and CB sizes; part of the program cache key."""
    return (
        INPUT_DEPTH,
        ROWS_DEPTH,
        BLOCK_W,
        BLOCK_W_CHOICES,
        (READ_CYCLES, WRITE_CYCLES, TILE_CYCLES, UNIT_CYCLES),
        READ_ALT_NOC_SHARE,
        BANK_MAJOR,
        BANK_RUN_ROUNDS,
        BANK_STAGGER,
    )


def _unit_geometry(W, block_w):
    """(block_w, units_per_row, tail_w, kw, kw_tail) for units of block_w pages; the whole row when W <= block_w."""
    if W <= block_w:
        kw = _div_up(W, TILE_ROWS)
        return W, 1, W, kw, kw
    units_per_row = _div_up(W, block_w)
    tail_w = W - (units_per_row - 1) * block_w
    return block_w, units_per_row, tail_w, _div_up(block_w, TILE_ROWS), _div_up(tail_w, TILE_ROWS)


def _l1_bytes(kw, ct, depth):
    """cb_input_slots + cb_output_rows + cb_tiles of one core."""
    return depth * kw * TILE_ROWS * ct * SLOT_ROW_BYTES + ROWS_DEPTH * kw * TILE_BYTES + kw * ct * TILE_BYTES


def _choose_block_w(N, H, W, C, num_cores, fits):
    """The unit width minimizing the busiest core's modelled time among the widths whose unit fits: its
    units at the pace of the slowest stage (read issue, compute, write issue), plus one unit's other two
    stages, the pipeline fill and drain that one unit per core cannot overlap. Ties go wider."""
    ct = _div_up(C, TILE_ROWS)
    best = None
    for candidate in BLOCK_W_CHOICES:
        block_w, units_per_row, _, kw, _ = _unit_geometry(W, candidate)
        if not fits(kw):
            continue
        units = N * H * units_per_row
        per_core = _div_up(units, min(units, num_cores))
        stages = (block_w * READ_CYCLES + UNIT_CYCLES, C * WRITE_CYCLES, kw * ct * TILE_CYCLES)
        cost = per_core * max(stages) + sum(stages) - max(stages)
        if best is None or cost <= best[0]:
            best = (cost, block_w)
        if block_w < candidate:
            break
    return None if best is None else best[1]


def build_plan(input_tensor, canonical_dims) -> NhwcRmPlan:
    assert tuple(canonical_dims) == DIMS, f"regime {REGIME} serves dims {DIMS}"
    assert INPUT_DEPTH >= 1 and ROWS_DEPTH >= 1
    N, H, W, C = (int(d) for d in input_tensor.shape)
    assert N * H * W == input_tensor.buffer_num_pages()

    element_bytes = input_tensor.element_size()
    in_page_bytes = input_tensor.buffer_page_size()
    assert element_bytes == 2 and in_page_bytes == C * element_bytes
    ct = _div_up(C, TILE_ROWS)
    slot_bytes = ct * SLOT_ROW_BYTES
    out_page_bytes = W * element_bytes

    buffer_type = input_tensor.memory_config().buffer_type
    # DRAM input: the shared reader (bank-major, 2-NoC split) lands pages at their bank stride, which is
    # the tilize row there (a DRAM page is padded to 64 B multiples). Measured on p44 in DRAM: 41.9 us,
    # against 63.9 us for the page-order reader. L1 input: the page-order reader with one push per
    # w-tile (11.6 us on p44, the shared reader 13.0 us); L1 pages are only 16 B-aligned, so the shared
    # reader's slots are the tilize row only when C is a multiple of 32.
    shared_reader = buffer_type == ttnn.BufferType.DRAM
    if shared_reader:
        assert input_tensor.buffer_aligned_page_size() == slot_bytes

    out_alignment = ttnn.get_l1_alignment() if buffer_type == ttnn.BufferType.L1 else ttnn.get_dram_alignment()
    out_bank_page_bytes = _div_up(out_page_bytes, out_alignment) * out_alignment
    out_l1 = out_l1_bytes_per_core(buffer_type, N * C * H, out_bank_page_bytes, input_tensor.device())
    budget = L1_OP_BUDGET_BYTES - out_l1

    grid = worker_grid(input_tensor.device())
    granule = ttnn.get_l1_alignment() // element_bytes
    if BLOCK_W is None:
        block_w = _choose_block_w(N, H, W, C, grid.x * grid.y, lambda kw: _l1_bytes(kw, ct, 1) <= budget)
    else:
        block_w = W if BLOCK_W >= W else BLOCK_W
        assert block_w == W or block_w % granule == 0, f"BLOCK_W must be >= W or a multiple of {granule}"
    if block_w is None or _unit_geometry(W, block_w)[3] > MAX_UNIT_TILES:
        raise RuntimeError(
            f"g6_permute: shape {[N, H, W, C]} leaves no room for one (n, h, w-block) unit within "
            f"{L1_OP_BUDGET_BYTES} B of L1 per core (output pages take {out_l1} B)"
        )
    block_w, units_per_row, tail_w, kw, kw_tail = _unit_geometry(W, block_w)

    tensor_units = N * H * units_per_row
    num_cores, all_cores, units_g1, units_g2, assignment = split_units(grid, tensor_units)
    input_depth = min(INPUT_DEPTH, units_g1)
    while input_depth > 1 and _l1_bytes(kw, ct, input_depth) > budget:
        input_depth -= 1
    l1_bytes = _l1_bytes(kw, ct, input_depth)
    if l1_bytes > budget:
        raise RuntimeError(
            f"g6_permute: shape {[N, H, W, C]} leaves no room for one (n, h, w-block) unit within "
            f"{L1_OP_BUDGET_BYTES} B of L1 per core (output pages take {out_l1} B, one unit {l1_bytes} B)"
        )

    alt_share, bank_major, num_banks, chunk_pages = read_plan(
        input_tensor, block_w, READ_ALT_NOC_SHARE, BANK_MAJOR, BANK_RUN_ROUNDS
    )
    if not shared_reader:
        alt_share, bank_major = Fraction(0), False

    core_roles = {
        f"{c.x},{c.y}": f"{REGIME} reader+compute+writer units (n, h, w-block) [{s}, {s + n})" for c, s, n in assignment
    }

    return NhwcRmPlan(
        regime=REGIME,
        num_cores=num_cores,
        tensor_units=tensor_units,
        units_per_core_g1=units_g1,
        units_per_core_g2=units_g2,
        block_w=block_w,
        tail_w=tail_w,
        units_per_row=units_per_row,
        kw=kw,
        kw_tail=kw_tail,
        ct=ct,
        input_depth=input_depth,
        rows_depth=ROWS_DEPTH,
        shared_reader=shared_reader,
        core_roles=core_roles,
        shape=(N, H, W, C),
        in_page_bytes=in_page_bytes,
        slot_bytes=slot_bytes,
        out_page_bytes=out_page_bytes,
        out_l1_bytes_per_core=out_l1,
        l1_bytes_per_core=l1_bytes,
        read_alt_share=alt_share,
        num_banks=num_banks,
        bank_stagger=BANK_STAGGER,
        bank_major=bank_major,
        chunk_pages=chunk_pages,
        all_cores=all_cores,
        assignment=assignment,
    )


def _unit_start(plan: NhwcRmPlan, unit):
    """(first input page, wb, h, p_out(n, c = 0, h)) of unit `unit`."""
    _, H, W, C = plan.shape
    nh, wb = divmod(unit, plan.units_per_row)
    n, h = divmod(nh, H)
    return nh * W + wb * plan.block_w, wb, h, n * C * H + h


def create_program_descriptor(plan: NhwcRmPlan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    """Descriptor for one plan. Buffer addresses are common runtime args (index 0) so a cached
    descriptor is re-dispatched by patching them; everything else is fixed per plan."""
    _, H, W, C = plan.shape
    assert output_tensor.buffer_page_size() == plan.out_page_bytes
    w_tile_bytes = TILE_ROWS * plan.slot_bytes
    # The shared reader pushes one page per unit, the page-order reader one page per w-tile.
    in_page_tiles = plan.kw if plan.shared_reader else 1

    def cb(index, pages, page_bytes):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint16, page_size=page_bytes)
            ],
        )

    cbs = [
        cb(CB_INPUT_SLOTS, plan.input_depth * plan.kw // in_page_tiles, in_page_tiles * w_tile_bytes),
        cb(CB_OUTPUT_ROWS, plan.rows_depth, plan.kw * TILE_BYTES),
        cb(CB_TILES, plan.kw * plan.ct, TILE_BYTES),
    ]

    starts = [(core, *_unit_start(plan, start), count) for core, start, count in plan.assignment]

    if plan.shared_reader:
        reader = reader_descriptor(
            plan,
            input_tensor,
            1,
            plan.block_w,
            plan.units_per_row,
            plan.tail_w,
            [(core, first_page, count, wb) for core, first_page, wb, _, _, count in starts],
        )
    else:
        reader_rt_args = ttnn.RuntimeArgs()
        for core, first_page, wb, _, _, count in starts:
            reader_rt_args[core.x][core.y] = [first_page, count, wb]
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_permute_nhwc_rm_reader.cpp"),
            core_ranges=plan.all_cores,
            compile_time_args=[
                CB_INPUT_SLOTS,
                plan.slot_bytes,
                plan.in_page_bytes,
                plan.kw,
                plan.units_per_row,
                plan.block_w,
                plan.tail_w,
                plan.input_depth * plan.kw,
                input_tensor.buffer_aligned_page_size(),
                plan.num_banks,
            ],
            runtime_args=reader_rt_args,
            common_runtime_args=[input_tensor.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )

    compute_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core, _, wb, h, page, count in starts:
        compute_rt_args[core.x][core.y] = [count, wb]
        writer_rt_args[core.x][core.y] = [count, wb, h, page]

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_nhwc_rm_compute.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=[
            CB_INPUT_SLOTS,
            CB_OUTPUT_ROWS,
            CB_TILES,
            plan.ct,
            plan.kw,
            plan.kw_tail,
            plan.units_per_row,
            in_page_tiles,
        ],
        runtime_args=compute_rt_args,
        # Values only move, as opaque UInt16: fidelity and approximation are moot. 16-bit half-sync DEST
        # holds a unit's kw <= 8 tiles.
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False, math_approx_mode=False),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_nhwc_rm_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=[
            CB_OUTPUT_ROWS,
            C,
            plan.ct,
            H,
            plan.units_per_row,
            plan.block_w,
            plan.tail_w,
            plan.kw,
            plan.kw_tail,
            plan.out_page_bytes,
            int(output_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM),
            ttnn.get_memory_view(output_tensor.device(), output_tensor.memory_config().buffer_type).num_banks,
            output_tensor.buffer_aligned_page_size(),
        ],
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=dm_config(plan, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, ttnn.WriterConfigDescriptor),
    )

    kernels = [None, None, None]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    kernels[COMPUTE_KERNEL_INDEX] = compute
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
