# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute, regimes blocks_rm_gather and blocks_rm_bank_coalesced: plan (unit split, block knobs,
read mode) and ProgramDescriptor.

ROW_MAJOR x [N, C, H, W, I, J], dims (0, 1, 4, 2, 5, 3): out[n, c, i, h, j, w] = x[n, c, h, w, i, j].
Work unit u = (n * C + c) * H + h. Its input is the W * I consecutive pages
[u * W * I, (u + 1) * W * I) of J values; its output is the I * J pages
p_out = ((n * C + c) * I + i) * H * J + h * J + j of W values. A core owns a contiguous unit range
and cuts it into blocks of block_h units; the reader lands each input page in its own slot and the
writer gathers the [W, I * J] values of a unit into I * J output rows in L1, then writes them.
blocks_rm_bank_coalesced differs in the read only: the pages of one bank are consecutive slots there,
so the reader fetches each bank's pages of a block in one read of whole slots and copies the payloads
into page order locally. In both regimes the writer groups its writes on DRAM output the same way: the
output pages of one bank for consecutive h are consecutive slots, written in one transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import ttnn

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

REGIME = "blocks_rm_gather"
REGIME_COALESCED = "blocks_rm_bank_coalesced"
DIMS = (0, 1, 4, 2, 5, 3)

CB_INPUT_SLOTS = 0
CB_OUTPUT_ROWS = 1
CB_BANK_STAGING = 2

# Largest single NoC packet (NOC_MAX_BURST_SIZE on Blackhole).
NOC_MAX_BURST_BYTES = 16384

# Buffer-depth knob of cb_input_slots; the allocated depth is min(INPUT_DEPTH, max_blocks_per_core).
INPUT_DEPTH = 2
# Input pages a block aims for; block_h = ceil(BLOCK_PAGES / pages_per_unit) under the L1 and core caps.
# A small block hides the gather of every block but the last under the reads: on FLUX_74 the whole
# per-core assignment as one block leaves a 55 us gather tail. Below ~256 pages, L1 input pays for the
# extra read barriers (FLUX_74: 152 us at 64 pages, 141 us at 256); DRAM input is flat up to 256 and
# slows at 512. None makes the whole per-core assignment one block.
BLOCK_PAGES = 256
# Tuning cap on block_h on top of BLOCK_PAGES and the L1 cap.
BLOCK_H_LIMIT = None
# DRAM input only. Every input page is one request at the DRAM port it enters, and a port retires them at
# a fixed rate; each NoC enters a bank through its own port. This is the share of the reader's page
# rounds issued on the second NoC, a power-of-two fraction below 1. A nonzero share puts both kernels in
# dynamic NoC mode. L1 input stays on one NoC: there the split only adds NoC contention.
READ_ALT_NOC_SHARE = Fraction(1, 2)
# DRAM input only: issue each block bank by bank (one set_state per bank and NoC) instead of in page order.
BANK_MAJOR = True
# Bank rounds (num_banks pages each) per bank-major chunk, which bounds a core's run of requests to one
# port; None makes the whole block one chunk.
BANK_RUN_ROUNDS = 8
# Rotate each core's per-block issue order by (core index mod banks) pages, or banks when bank-major.
BANK_STAGGER = False

# "gather" (one read per page), "coalesced" (one read of whole slots per bank run), or "auto": the read
# mode measured faster for the input's placement (COALESCED_BUFFERS), when a coalesced block fits L1.
# Coalesced wins in DRAM and in L1 on FLUX_73 and FLUX_74; in DRAM a page read costs a whole request at
# a DRAM port either way, so whole 64 B slots cost little more than their 4 B payloads.
READ_MODE = "auto"
COALESCED_BUFFERS = (ttnn.BufferType.DRAM, ttnn.BufferType.L1)
# Coalesced mode: input pages a block aims for, per input placement (or one int for every placement):
# block_h = ceil(block pages / pages_per_unit) under the L1 and core caps. Each bank's read is about
# block pages / num_banks slots. DRAM has 8 banks and L1 one per worker, so L1 needs larger blocks for
# the same read length: FLUX_74 in L1 takes 188 us at 256 pages, 68 us at 2048, 92 us at 4096.
COALESCED_BLOCK_PAGES = {ttnn.BufferType.DRAM: 256, ttnn.BufferType.L1: 2048}
# Coalesced mode: stages of cb_bank_staging; the allocated depth is min(STAGING_DEPTH, max_blocks_per_core).
# 2 issues the next block's reads before the reader copies the current one. That loses once the copies
# are batched (FLUX_74 DRAM 116 us against 89 us at 1): every core then keeps reads in flight without a
# break, and the NoC links into the cores far from the DRAM columns congest, stalling their reads and
# writes. One stage lets each core's reads pause during its copy.
STAGING_DEPTH = 1
# Coalesced mode, DRAM input only: share of each bank run read on the second NoC (as READ_ALT_NOC_SHARE).
COALESCED_ALT_NOC_SHARE = Fraction(1, 2)
# Coalesced mode: rotate each core's bank order by (core index mod banks), as BANK_STAGGER.
COALESCED_BANK_STAGGER = False
# Coalesced mode: payload words the reader loads before storing them when it copies the staged slots
# into cb_input_slots, in whole pages (1 copies page by page). FLUX_74: 4356 -> 1030 cycles per block.
COMPACT_BATCH_WORDS = 8
# Page pairs whose loads the writer's gather issues before storing them (1 walks pair by pair).
# FLUX_74: 3558 -> 2144 cycles per block.
GATHER_PAIR_BATCH = 8
# Output placements in WRITE_GROUP_BUFFERS, and J must divide the bank count: up to this many
# consecutive units of one (n, c) form a write group, whose output rows the writer stages bank by bank
# and writes with one transaction per (i, bank) instead of one per row. None writes every row on its own.
# FLUX_74 DRAM: 90.5 us ungrouped, 84.7 at 32 units, 83.4 at 64, 82.2 at 128. Into L1 output it loses
# (FLUX_74 78.1 against 67.7 us): L1 banks take small writes cheaply.
WRITE_GROUP_UNITS = 64
WRITE_GROUP_BUFFERS = (ttnn.BufferType.DRAM,)
# Write-group buffers: with 2 the gather fills one while the other's writes drain.
WRITE_GROUP_DEPTH = 2


@dataclass
class BlocksPlan:
    regime: str
    num_cores: int
    tensor_units: int
    units_per_core_g1: int
    units_per_core_g2: int
    block_h: int
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
    out_row_bytes: int = 0
    read_alt_share: Fraction = Fraction(0)
    num_banks: int = 1
    bank_stagger: bool = False
    bank_major: bool = False
    chunk_pages: int = 0
    # Byte distance of consecutive pages in cb_input_slots: slot_bytes, or the payload rounded up to a
    # word when the reader compacts the coalesced slots.
    page_stride: int = 0
    coalesced: bool = False
    staging_rounds: int = 0
    staging_depth: int = 0
    # Grouped writes: units per group (0 writes each row on its own), buffers, slots per bank region,
    # and the output's bank stride.
    write_group_units: int = 0
    write_group_depth: int = 0
    group_slots: int = 0
    out_slot_bytes: int = 0
    all_cores: object = None
    assignment: list = field(default_factory=list)  # [(CoreCoord, start_unit, num_units)]


def knob_key():
    """Host knobs that reach this regime's CT args; part of the program cache key."""
    return (
        INPUT_DEPTH,
        BLOCK_PAGES,
        BLOCK_H_LIMIT,
        READ_ALT_NOC_SHARE,
        BANK_MAJOR,
        BANK_RUN_ROUNDS,
        BANK_STAGGER,
        READ_MODE,
        COALESCED_BUFFERS,
        _hashable(COALESCED_BLOCK_PAGES),
        STAGING_DEPTH,
        COALESCED_ALT_NOC_SHARE,
        COALESCED_BANK_STAGGER,
        COMPACT_BATCH_WORDS,
        GATHER_PAIR_BATCH,
        WRITE_GROUP_UNITS,
        WRITE_GROUP_BUFFERS,
        WRITE_GROUP_DEPTH,
    )


def _hashable(knob):
    return tuple(sorted(knob.items(), key=str)) if isinstance(knob, dict) else knob


def _per_placement(knob, buffer_type):
    return knob.get(buffer_type) if isinstance(knob, dict) else knob


def _align_up(value, alignment):
    return _div_up(value, alignment) * alignment


def build_plan(input_tensor, canonical_dims) -> BlocksPlan:
    assert tuple(canonical_dims) == DIMS, f"regime {REGIME} serves dims {DIMS}"
    assert INPUT_DEPTH >= 1
    N, C, H, W, I, J = (int(d) for d in input_tensor.shape)
    tensor_units = N * C * H
    pages_per_unit = W * I
    assert tensor_units * pages_per_unit == input_tensor.buffer_num_pages()

    element_bytes = input_tensor.element_size()
    in_page_bytes = input_tensor.buffer_page_size()
    assert in_page_bytes == J * element_bytes
    # A page's bank stride: every page lands at the same offset mod the DRAM read alignment as its source.
    slot_bytes = input_tensor.buffer_aligned_page_size()
    out_page_bytes = W * element_bytes
    # A staging row is a NoC write source: aligned like the destination page start.
    out_row_bytes = _align_up(out_page_bytes, ttnn.get_l1_alignment())

    buffer_type = input_tensor.memory_config().buffer_type
    out_bank_page_bytes = _align_up(
        out_page_bytes,
        ttnn.get_l1_alignment() if buffer_type == ttnn.BufferType.L1 else ttnn.get_dram_alignment(),
    )
    out_l1 = out_l1_bytes_per_core(buffer_type, tensor_units * I * J, out_bank_page_bytes, input_tensor.device())

    unit_row_bytes = I * J * out_row_bytes
    num_banks = ttnn.get_memory_view(input_tensor.device(), buffer_type).num_banks
    budget = L1_OP_BUDGET_BYTES - out_l1

    num_cores, all_cores, units_g1, units_g2, assignment = split_units(worker_grid(input_tensor.device()), tensor_units)

    # Grouped writes stage the output rows per group instead of per block.
    write_group_units, write_group_depth, group_slots = _write_groups(
        buffer_type, num_banks, H, J, out_bank_page_bytes, units_g1
    )
    if write_group_units:
        budget -= write_group_depth * I * num_banks * group_slots * out_bank_page_bytes
        unit_row_bytes = 0

    # Coalesced: cb_input_slots holds word-aligned payloads, and each stage of cb_bank_staging holds a
    # block's whole slots plus up to two partial bank rounds: NB * slot * rounds(n) <= slot * (n + 2 NB).
    page_stride = _align_up(in_page_bytes, 4)
    coalesced_cap = (budget - STAGING_DEPTH * slot_bytes * 2 * num_banks) // (
        INPUT_DEPTH * pages_per_unit * page_stride + unit_row_bytes + STAGING_DEPTH * pages_per_unit * slot_bytes
    )
    coalesced = _coalesced_reads(buffer_type, coalesced_cap)
    if coalesced:
        block_cap, block_pages_knob = coalesced_cap, _per_placement(COALESCED_BLOCK_PAGES, buffer_type)
    else:
        page_stride = slot_bytes
        block_cap = budget // (INPUT_DEPTH * pages_per_unit * slot_bytes + unit_row_bytes)
        block_pages_knob = BLOCK_PAGES
    if block_cap < 1:
        raise RuntimeError(
            f"g6_permute: shape {[N, C, H, W, I, J]} leaves no room for one (n, c, h) block within "
            f"{L1_OP_BUDGET_BYTES} B of L1 per core (output pages take {out_l1} B, one unit "
            f"{INPUT_DEPTH} x {pages_per_unit * slot_bytes} B of input slots + {unit_row_bytes} B of output rows)"
        )

    block_h = min(units_g1, block_cap)
    if block_pages_knob is not None:
        block_h = min(block_h, _div_up(block_pages_knob, pages_per_unit))
    if BLOCK_H_LIMIT is not None:
        block_h = min(block_h, BLOCK_H_LIMIT)
    assert block_h >= 1

    block_pages = block_h * pages_per_unit
    alt_share, bank_major, num_banks, chunk_pages = read_plan(
        input_tensor,
        block_pages,
        COALESCED_ALT_NOC_SHARE if coalesced else READ_ALT_NOC_SHARE,
        BANK_MAJOR and not coalesced,
        BANK_RUN_ROUNDS,
    )
    max_blocks_per_core = _div_up(units_g1, block_h)

    staging_rounds = staging_depth = 0
    if coalesced:
        # Every block starts at a unit boundary: a whole bank round when pages_per_unit is one.
        first_bank_max = 0 if pages_per_unit % num_banks == 0 else num_banks - 1
        staging_rounds = (first_bank_max + block_pages - 1) // num_banks + 1
        staging_depth = min(STAGING_DEPTH, max_blocks_per_core)

    regime = REGIME_COALESCED if coalesced else REGIME
    core_roles = {f"{c.x},{c.y}": f"{regime} reader+writer units (n, c, h) [{s}, {s + n})" for c, s, n in assignment}

    return BlocksPlan(
        regime=regime,
        num_cores=num_cores,
        tensor_units=tensor_units,
        units_per_core_g1=units_g1,
        units_per_core_g2=units_g2,
        block_h=block_h,
        block_cap=block_cap,
        input_depth=min(INPUT_DEPTH, max_blocks_per_core),
        max_blocks_per_core=max_blocks_per_core,
        num_blocks_g1=_div_up(units_g1, block_h),
        num_blocks_g2=_div_up(units_g2, block_h),
        core_roles=core_roles,
        shape=(N, C, H, W, I, J),
        in_page_bytes=in_page_bytes,
        slot_bytes=slot_bytes,
        out_page_bytes=out_page_bytes,
        out_row_bytes=out_row_bytes,
        read_alt_share=alt_share,
        num_banks=num_banks,
        bank_stagger=COALESCED_BANK_STAGGER if coalesced else BANK_STAGGER,
        bank_major=bank_major,
        chunk_pages=chunk_pages,
        page_stride=page_stride,
        coalesced=coalesced,
        staging_rounds=staging_rounds,
        staging_depth=staging_depth,
        write_group_units=write_group_units,
        write_group_depth=write_group_depth,
        group_slots=group_slots,
        out_slot_bytes=out_bank_page_bytes,
        all_cores=all_cores,
        assignment=assignment,
    )


def _write_groups(buffer_type, num_banks, H, J, out_slot_bytes, units_per_core):
    """(units per group, buffers, slots per bank region) of grouped writes, or (0, 0, 0). A group stays
    inside one (n, c), and one (run, bank) write is one NoC packet."""
    if WRITE_GROUP_UNITS is None or buffer_type not in WRITE_GROUP_BUFFERS or num_banks % J != 0:
        return 0, 0, 0
    units = min(WRITE_GROUP_UNITS, H, units_per_core)

    def slots(units):
        # A run starts at a bank that is a multiple of J, so at most at num_banks - J.
        return (num_banks - J + units * J - 1) // num_banks + 1

    while units > 1 and slots(units) * out_slot_bytes > NOC_MAX_BURST_BYTES:
        units -= 1
    if slots(units) * out_slot_bytes > NOC_MAX_BURST_BYTES:
        return 0, 0, 0
    depth = WRITE_GROUP_DEPTH if units < units_per_core else 1
    return units, depth, slots(units)


def _coalesced_reads(buffer_type, coalesced_cap):
    """The read mode: READ_MODE, or under "auto" the measured choice for the placement, when one
    coalesced unit fits the L1 budget."""
    if READ_MODE == "gather":
        return False
    if READ_MODE == "coalesced":
        return True
    assert READ_MODE == "auto", f"READ_MODE {READ_MODE!r}"
    return buffer_type in COALESCED_BUFFERS and coalesced_cap >= 1


def read_plan(input_tensor, block_pages, alt_noc_share, bank_major, bank_run_rounds):
    """The shared reader's issue plan for blocks of up to block_pages pages:
    (read_alt_share, bank_major, num_banks, chunk_pages). The 2-NoC split and the bank-major walk
    apply to DRAM input only."""
    buffer_type = input_tensor.memory_config().buffer_type
    dram_input = buffer_type == ttnn.BufferType.DRAM
    alt_share = Fraction(alt_noc_share) if dram_input else Fraction(0)
    assert 0 <= alt_share < 1 and alt_share.denominator & (alt_share.denominator - 1) == 0
    num_banks = ttnn.get_memory_view(input_tensor.device(), buffer_type).num_banks
    chunk_pages = block_pages if bank_run_rounds is None else min(block_pages, bank_run_rounds * num_banks)
    return alt_share, bank_major and dram_input, num_banks, _align_up(chunk_pages, num_banks)


def reader_descriptor(
    plan, input_tensor, block_units, pages_per_unit, units_per_row, tail_pages, core_starts, staging=None
):
    """The shared ROW_MAJOR reader (g6_permute_blocks_reader.cpp) for a plan carrying the read_plan
    fields. Blocks are block_units units of pages_per_unit pages, except the last unit of each row of
    units_per_row, which holds tail_pages. core_starts: [(core, start_page, num_units, row_pos)].
    staging: None for one read per page, or (staging_rounds, staging_depth, page_stride) for
    bank-coalesced reads through CB_BANK_STAGING."""
    coalesced_args = (
        [0, CB_BANK_STAGING, 0, 0, plan.slot_bytes, 1]
        if staging is None
        else [1, CB_BANK_STAGING, *staging, COMPACT_BATCH_WORDS]
    )
    ct_args = [
        CB_INPUT_SLOTS,
        block_units,
        pages_per_unit,
        plan.in_page_bytes,
        plan.slot_bytes,
        plan.read_alt_share.numerator,
        plan.read_alt_share.denominator,
        plan.num_banks,
        int(plan.bank_major),
        plan.chunk_pages,
        units_per_row,
        tail_pages,
        *coalesced_args,
        *ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
    ]
    rt_args = ttnn.RuntimeArgs()
    for core_idx, (core, start_page, count, row_pos) in enumerate(core_starts):
        stagger = core_idx % plan.num_banks if plan.bank_stagger else 0
        rt_args[core.x][core.y] = [start_page, count, stagger, row_pos]
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_blocks_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=dm_config(plan, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0, ttnn.ReaderConfigDescriptor),
    )


def _first_output_page(plan: BlocksPlan, unit):
    """(h, p_out(n, c, i=0, h, j=0)) of unit `unit`; the writer advances both without dividing."""
    _, _, H, _, I, J = plan.shape
    nc, h = divmod(unit, H)
    return h, nc * I * H * J + h * J


def dm_config(plan, processor, noc, dedicated_config):
    """The default reader/writer config, or the same RISC and NoC in dynamic NoC mode when the reader
    also issues on the second NoC (one core runs both kernels in one mode)."""
    if plan.read_alt_share == 0:
        return dedicated_config()
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)


def create_program_descriptor(plan: BlocksPlan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    """Descriptor for one plan. Buffer addresses are common runtime args (index 0) so a cached
    descriptor is re-dispatched by patching them; everything else is fixed per plan."""
    N, C, H, W, I, J = plan.shape
    assert output_tensor.buffer_page_size() == plan.out_page_bytes
    unit_slot_bytes = W * I * plan.page_stride
    if plan.write_group_units:
        assert output_tensor.buffer_aligned_page_size() == plan.out_slot_bytes
        rows_page_bytes = I * plan.num_banks * plan.group_slots * plan.out_slot_bytes
        rows_bytes = plan.write_group_depth * rows_page_bytes
    else:
        rows_page_bytes = I * J * plan.out_row_bytes
        rows_bytes = plan.block_h * rows_page_bytes

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan.input_depth * plan.block_h * unit_slot_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_SLOTS, data_format=input_tensor.dtype, page_size=unit_slot_bytes
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=rows_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_ROWS, data_format=input_tensor.dtype, page_size=rows_page_bytes
                )
            ],
        ),
    ]
    staging = None
    if plan.coalesced:
        stage_bytes = plan.num_banks * plan.staging_rounds * plan.slot_bytes
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan.staging_depth * stage_bytes,
                core_ranges=plan.all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BANK_STAGING, data_format=input_tensor.dtype, page_size=stage_bytes
                    )
                ],
            )
        )
        staging = (plan.staging_rounds, plan.staging_depth, plan.page_stride)

    writer_ct_args = [
        CB_INPUT_SLOTS,
        CB_OUTPUT_ROWS,
        plan.block_h,
        H,
        W,
        I,
        J,
        plan.page_stride,
        plan.out_page_bytes,
        plan.out_row_bytes,
        GATHER_PAIR_BATCH,
        plan.write_group_units,
        plan.write_group_depth,
        plan.num_banks,
        plan.out_slot_bytes,
        plan.group_slots,
        *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
    ]

    writer_rt_args = ttnn.RuntimeArgs()
    for core, start, count in plan.assignment:
        writer_rt_args[core.x][core.y] = [count, *_first_output_page(plan, start)]

    # One unit per row: every unit is a full W * I pages.
    reader = reader_descriptor(
        plan,
        input_tensor,
        plan.block_h,
        W * I,
        1,
        W * I,
        [(core, start * W * I, count, 0) for core, start, count in plan.assignment],
        staging,
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_blocks_writer.cpp"),
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
