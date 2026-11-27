# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute, regime blocks_rm_gather: plan (unit split, block knobs) and ProgramDescriptor.

ROW_MAJOR x [N, C, H, W, I, J], dims (0, 1, 4, 2, 5, 3): out[n, c, i, h, j, w] = x[n, c, h, w, i, j].
Work unit u = (n * C + c) * H + h. Its input is the W * I consecutive pages
[u * W * I, (u + 1) * W * I) of J values; its output is the I * J pages
p_out = ((n * C + c) * I + i) * H * J + h * J + j of W values. A core owns a contiguous unit range
and cuts it into blocks of block_h units; the reader lands each input page in its own slot and the
writer gathers the [W, I * J] values of a unit into I * J output rows in L1, then writes them.
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
DIMS = (0, 1, 4, 2, 5, 3)

CB_INPUT_SLOTS = 0
CB_OUTPUT_ROWS = 1

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
    all_cores: object = None
    assignment: list = field(default_factory=list)  # [(CoreCoord, start_unit, num_units)]


def knob_key():
    """Host knobs that reach this regime's CT args; part of the program cache key."""
    return (INPUT_DEPTH, BLOCK_PAGES, BLOCK_H_LIMIT, READ_ALT_NOC_SHARE, BANK_MAJOR, BANK_RUN_ROUNDS, BANK_STAGGER)


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

    unit_slot_bytes = pages_per_unit * slot_bytes
    unit_row_bytes = I * J * out_row_bytes
    block_cap = (L1_OP_BUDGET_BYTES - out_l1) // (INPUT_DEPTH * unit_slot_bytes + unit_row_bytes)
    if block_cap < 1:
        raise RuntimeError(
            f"g6_permute: shape {[N, C, H, W, I, J]} leaves no room for one (n, c, h) block within "
            f"{L1_OP_BUDGET_BYTES} B of L1 per core (output pages take {out_l1} B, one unit "
            f"{INPUT_DEPTH} x {unit_slot_bytes} B of input slots + {unit_row_bytes} B of output rows)"
        )

    num_cores, all_cores, units_g1, units_g2, assignment = split_units(worker_grid(input_tensor.device()), tensor_units)

    block_h = min(units_g1, block_cap)
    if BLOCK_PAGES is not None:
        block_h = min(block_h, _div_up(BLOCK_PAGES, pages_per_unit))
    if BLOCK_H_LIMIT is not None:
        block_h = min(block_h, BLOCK_H_LIMIT)
    assert block_h >= 1

    dram_input = buffer_type == ttnn.BufferType.DRAM
    alt_share = Fraction(READ_ALT_NOC_SHARE) if dram_input else Fraction(0)
    assert 0 <= alt_share < 1 and alt_share.denominator & (alt_share.denominator - 1) == 0
    bank_major = BANK_MAJOR and dram_input
    num_banks = ttnn.get_memory_view(input_tensor.device(), buffer_type).num_banks
    block_pages = block_h * pages_per_unit
    chunk_pages = block_pages if BANK_RUN_ROUNDS is None else min(block_pages, BANK_RUN_ROUNDS * num_banks)
    chunk_pages = _align_up(chunk_pages, num_banks)
    max_blocks_per_core = _div_up(units_g1, block_h)

    core_roles = {f"{c.x},{c.y}": f"{REGIME} reader+writer units (n, c, h) [{s}, {s + n})" for c, s, n in assignment}

    return BlocksPlan(
        regime=REGIME,
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
        bank_stagger=BANK_STAGGER,
        bank_major=bank_major,
        chunk_pages=chunk_pages,
        all_cores=all_cores,
        assignment=assignment,
    )


def _first_output_page(plan: BlocksPlan, unit):
    """(h, p_out(n, c, i=0, h, j=0)) of unit `unit`; the writer advances both without dividing."""
    _, _, H, _, I, J = plan.shape
    nc, h = divmod(unit, H)
    return h, nc * I * H * J + h * J


def _dm_config(plan: BlocksPlan, processor, noc, dedicated_config):
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
    unit_slot_bytes = W * I * plan.slot_bytes
    unit_row_bytes = I * J * plan.out_row_bytes

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
            total_size=plan.block_h * unit_row_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_ROWS, data_format=input_tensor.dtype, page_size=unit_row_bytes
                )
            ],
        ),
    ]

    reader_ct_args = [
        CB_INPUT_SLOTS,
        plan.block_h,
        W * I,
        plan.in_page_bytes,
        plan.slot_bytes,
        plan.read_alt_share.numerator,
        plan.read_alt_share.denominator,
        plan.num_banks,
        int(plan.bank_major),
        plan.chunk_pages,
        *ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
    ]
    writer_ct_args = [
        CB_INPUT_SLOTS,
        CB_OUTPUT_ROWS,
        plan.block_h,
        H,
        W,
        I,
        J,
        plan.slot_bytes,
        plan.out_page_bytes,
        plan.out_row_bytes,
        *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
    ]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core_idx, (core, start, count) in enumerate(plan.assignment):
        stagger = core_idx % plan.num_banks if plan.bank_stagger else 0
        reader_rt_args[core.x][core.y] = [start, count, stagger]
        writer_rt_args[core.x][core.y] = [count, *_first_output_page(plan, start)]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_blocks_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=_dm_config(plan, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0, ttnn.ReaderConfigDescriptor),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_blocks_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=_dm_config(plan, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, ttnn.WriterConfigDescriptor),
    )
    kernels = [None, None]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
