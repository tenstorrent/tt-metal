# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute, regime nhwc_rm_transpose: plan (unit split, block knobs) and ProgramDescriptor.

ROW_MAJOR x [N, H, W, C], dims (0, 3, 1, 2): out[n, c, h, w] = x[n, h, w, c].
Input page (n, h, w) holds the C values of one pixel; output page (n, c, h) holds the W values of one
channel row. Each row (n, h) of W input pages is cut into units of block_w pages (the last unit of a
row holds the tail_w left over). Unit (n, h, wb) is the input pages [(n * H + h) * W + wb * block_w,
+ width), and it writes the C partial output rows (n, c, h), values [wb * block_w, + width), at byte
offset wb * block_w * 2. Consecutive units are consecutive input pages, so a core's contiguous unit
range is one page range; the reader is blocks_rm_gather's, the writer transposes [width, C] into
[C, width] in L1 and writes the C rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import ttnn

from .g6_permute_blocks_rm_gather import (
    CB_INPUT_SLOTS,
    CB_OUTPUT_ROWS,
    dm_config,
    read_plan,
    reader_descriptor,
)
from .g6_permute_program_descriptor import (
    KERNEL_DIR,
    L1_OP_BUDGET_BYTES,
    READER_KERNEL_INDEX,
    WRITER_KERNEL_INDEX,
    _div_up,
    out_l1_bytes_per_core,
    split_units,
    worker_grid,
)

REGIME = "nhwc_rm_transpose"
DIMS = (0, 3, 1, 2)

# Buffer-depth knob of cb_input_slots; the allocated depth is min(INPUT_DEPTH, max_blocks_per_core).
INPUT_DEPTH = 2
# Unit width in input pages (a multiple of the write granule, or >= W for whole rows); None picks the
# width that minimizes the busiest core's modelled time (_choose_block_w).
BLOCK_W = None
# Input pages a block aims for: block_units = ceil(BLOCK_PAGES / block_w) under the L1 and core caps.
BLOCK_PAGES = 256
# Per-core time model of _choose_block_w, in ns: one input page (its read issue plus the transpose
# of its C values, GATHER_NS each) and one partial-row write issue.
READ_NS = 28
GATHER_NS = 6
WRITE_NS = 30
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
    block_units: int
    block_cap: int
    input_depth: int
    max_blocks_per_core: int
    num_blocks_g1: int
    num_blocks_g2: int
    core_roles: dict
    shape: tuple = ()
    in_page_bytes: int = 0
    slot_bytes: int = 0
    out_page_bytes: int = 0
    row_bytes: int = 0
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
    """Host knobs that reach this regime's CT args; part of the program cache key."""
    return (
        INPUT_DEPTH,
        BLOCK_W,
        BLOCK_PAGES,
        (READ_NS, GATHER_NS, WRITE_NS),
        READ_ALT_NOC_SHARE,
        BANK_MAJOR,
        BANK_RUN_ROUNDS,
        BANK_STAGGER,
    )


def _align_up(value, alignment):
    return _div_up(value, alignment) * alignment


def _unit_l1_bytes(block_w, C, slot_bytes, element_bytes):
    """(input slot bytes, staging row stride, staging bytes) of one unit."""
    row_bytes = _align_up(block_w * element_bytes, ttnn.get_l1_alignment())
    return block_w * slot_bytes, row_bytes, C * row_bytes


def _choose_block_w(N, H, W, C, num_cores, granule, fits):
    """The unit width minimizing the busiest core's modelled time: whole rows, or a multiple of the
    write granule below W. Ties go to the wider unit (fewer handshakes and writes)."""
    best = None
    for block_w in [W, *range(granule * ((W - 1) // granule), 0, -granule)]:
        if not fits(block_w):
            continue
        units = N * H * _div_up(W, block_w)
        cost = _div_up(units, min(units, num_cores)) * (block_w * (READ_NS + C * GATHER_NS) + C * WRITE_NS)
        if best is None or cost < best[0]:
            best = (cost, block_w)
    return None if best is None else best[1]


def build_plan(input_tensor, canonical_dims) -> NhwcRmPlan:
    assert tuple(canonical_dims) == DIMS, f"regime {REGIME} serves dims {DIMS}"
    assert INPUT_DEPTH >= 1
    N, H, W, C = (int(d) for d in input_tensor.shape)
    assert N * H * W == input_tensor.buffer_num_pages()

    element_bytes = input_tensor.element_size()
    in_page_bytes = input_tensor.buffer_page_size()
    assert in_page_bytes == C * element_bytes
    # A page's bank stride: every page lands at the same offset mod the read alignment as its source.
    slot_bytes = input_tensor.buffer_aligned_page_size()
    out_page_bytes = W * element_bytes
    # A partial row is written at byte offset wb * block_w * element_bytes: NoC writes need it
    # aligned like its 16 B-aligned staging row.
    granule = ttnn.get_l1_alignment() // element_bytes

    buffer_type = input_tensor.memory_config().buffer_type
    out_bank_page_bytes = _align_up(
        out_page_bytes,
        ttnn.get_l1_alignment() if buffer_type == ttnn.BufferType.L1 else ttnn.get_dram_alignment(),
    )
    out_l1 = out_l1_bytes_per_core(buffer_type, N * C * H, out_bank_page_bytes, input_tensor.device())

    def unit_cap(block_w):
        unit_slot_bytes, _, unit_row_bytes = _unit_l1_bytes(block_w, C, slot_bytes, element_bytes)
        return (L1_OP_BUDGET_BYTES - out_l1) // (INPUT_DEPTH * unit_slot_bytes + unit_row_bytes)

    grid = worker_grid(input_tensor.device())
    num_grid_cores = grid.x * grid.y
    if BLOCK_W is None:
        block_w = _choose_block_w(N, H, W, C, num_grid_cores, granule, lambda bw: unit_cap(bw) >= 1)
    else:
        block_w = W if BLOCK_W >= W else BLOCK_W
        assert block_w == W or block_w % granule == 0, f"BLOCK_W must be >= W or a multiple of {granule}"
    if block_w is None or unit_cap(block_w) < 1:
        raise RuntimeError(
            f"g6_permute: shape {[N, H, W, C]} leaves no room for one (n, h, w-block) unit within "
            f"{L1_OP_BUDGET_BYTES} B of L1 per core (output pages take {out_l1} B)"
        )
    block_cap = unit_cap(block_w)
    units_per_row = _div_up(W, block_w)
    tail_w = W - (units_per_row - 1) * block_w
    unit_slot_bytes, row_bytes, unit_row_bytes = _unit_l1_bytes(block_w, C, slot_bytes, element_bytes)

    tensor_units = N * H * units_per_row
    num_cores, all_cores, units_g1, units_g2, assignment = split_units(grid, tensor_units)

    block_units = min(units_g1, block_cap)
    if BLOCK_PAGES is not None:
        block_units = min(block_units, _div_up(BLOCK_PAGES, block_w))
    assert block_units >= 1
    max_blocks_per_core = _div_up(units_g1, block_units)
    input_depth = min(INPUT_DEPTH, max_blocks_per_core)

    alt_share, bank_major, num_banks, chunk_pages = read_plan(
        input_tensor, block_units * block_w, READ_ALT_NOC_SHARE, BANK_MAJOR, BANK_RUN_ROUNDS
    )

    core_roles = {
        f"{c.x},{c.y}": f"{REGIME} reader+writer units (n, h, w-block) [{s}, {s + n})" for c, s, n in assignment
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
        block_units=block_units,
        block_cap=block_cap,
        input_depth=input_depth,
        max_blocks_per_core=max_blocks_per_core,
        num_blocks_g1=_div_up(units_g1, block_units),
        num_blocks_g2=_div_up(units_g2, block_units),
        core_roles=core_roles,
        shape=(N, H, W, C),
        in_page_bytes=in_page_bytes,
        slot_bytes=slot_bytes,
        out_page_bytes=out_page_bytes,
        row_bytes=row_bytes,
        out_l1_bytes_per_core=out_l1,
        l1_bytes_per_core=input_depth * block_units * unit_slot_bytes + block_units * unit_row_bytes,
        read_alt_share=alt_share,
        num_banks=num_banks,
        bank_stagger=BANK_STAGGER,
        bank_major=bank_major,
        chunk_pages=chunk_pages,
        all_cores=all_cores,
        assignment=assignment,
    )


def _unit_start(plan: NhwcRmPlan, unit):
    """(first input page, writer cursor (wb, h, p_out(n, c = 0, h))) of unit `unit`."""
    _, H, W, C = plan.shape
    nh, wb = divmod(unit, plan.units_per_row)
    n, h = divmod(nh, H)
    return nh * W + wb * plan.block_w, (wb, h, n * C * H + h)


def create_program_descriptor(plan: NhwcRmPlan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    """Descriptor for one plan. Buffer addresses are common runtime args (index 0) so a cached
    descriptor is re-dispatched by patching them; everything else is fixed per plan."""
    _, H, W, C = plan.shape
    assert output_tensor.buffer_page_size() == plan.out_page_bytes
    unit_slot_bytes = plan.block_w * plan.slot_bytes
    unit_row_bytes = C * plan.row_bytes

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan.input_depth * plan.block_units * unit_slot_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_SLOTS, data_format=input_tensor.dtype, page_size=unit_slot_bytes
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=plan.block_units * unit_row_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_ROWS, data_format=input_tensor.dtype, page_size=unit_row_bytes
                )
            ],
        ),
    ]

    starts = [(core, *_unit_start(plan, start), count) for core, start, count in plan.assignment]
    reader = reader_descriptor(
        plan,
        input_tensor,
        plan.block_units,
        plan.block_w,
        plan.units_per_row,
        plan.tail_w,
        [(core, first_page, count, cursor[0]) for core, first_page, cursor, count in starts],
    )

    writer_ct_args = [
        CB_INPUT_SLOTS,
        CB_OUTPUT_ROWS,
        plan.block_units,
        H,
        C,
        plan.units_per_row,
        plan.block_w,
        plan.tail_w,
        plan.slot_bytes,
        plan.row_bytes,
        plan.out_page_bytes,
        *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
    ]
    writer_rt_args = ttnn.RuntimeArgs()
    for core, _, cursor, count in starts:
        writer_rt_args[core.x][core.y] = [count, *cursor]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_nhwc_rm_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=dm_config(plan, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, ttnn.WriterConfigDescriptor),
    )

    kernels = [None, None]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
