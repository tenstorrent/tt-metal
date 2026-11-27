# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros program descriptor: bank-affine bank-run zero writes (regime R1).

In an interleaved DRAM tensor page p lies in bank p % num_dram_banks at slot
p // num_dram_banks, and one bank's slots are contiguous. Each writer core owns
a set of `BANKS_PER_CORE` banks and one balanced slot range in that set, and
writes every bank's run of its slot range as contiguous NoC writes from a
core-local zero source. Every page is written exactly once; nothing is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = str(KERNEL_DIR / "g6_zeros_writer.cpp")

TILE_DIM = 32

# --- Block knobs (each defined once; every dependent quantity derives from them) ---

# Upper bound on the zero source and on one NoC write: NOC_MAX_BURST_SIZE on Blackhole.
ZERO_SOURCE_MAX_BYTES = 16384
# Writer cores per DRAM bank (max_writer_cores = num_dram_banks * this, capped by the grid). None = the
# whole compute grid. 1 measured fastest on all four production shapes: more writers per bank only add
# NoC and DRAM-endpoint contention.
WRITER_CORES_PER_BANK = 1
# Banks one core visits. 1 = bank-affine; num_dram_banks = every core walks every bank, staggered.
BANKS_PER_CORE = 1
# Slots of one bank written as one block. None = the core's whole slot range.
BLOCK_SLOTS = None

CB_ZERO_SOURCE = 0
# A byte scratch that never passes the packer: any non-block format holds it, whatever the output dtype.
ZERO_SOURCE_CB_FORMAT = ttnn.float32


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class CoreWork:
    core: tuple  # (x, y) logical worker coordinate
    bank_set_base: int
    first_bank_offset: int
    slot_begin: int
    slots: int
    block_slots: int


@dataclass(frozen=True)
class ZerosPlan:
    tensor_pages: int
    page_bytes: int
    grid: tuple  # (x, y)
    num_dram_banks: int
    banks_per_core: int
    num_writer_cores: int
    zero_source_bytes: int
    cores: tuple  # of CoreWork, row-major core order
    writer_riscv: str = "BRISC (RISCV_0)"
    writer_noc: str = "NOC_1"


def tensor_pages_of(shape) -> int:
    d0, d1, d2, d3 = shape
    return d0 * d1 * (d2 // TILE_DIM) * (d3 // TILE_DIM)


def page_bytes_of(dtype) -> int:
    return _ceil_div(ttnn.tile_size(dtype), ttnn.get_dram_alignment()) * ttnn.get_dram_alignment()


def device_geometry(device):
    grid = device.compute_with_storage_grid_size()
    dram = device.dram_grid_size()
    return (grid.x, grid.y), dram.x * dram.y


def make_plan(shape, dtype, grid, num_dram_banks) -> ZerosPlan:
    """Derive the per-core split from the shape, dtype, device geometry and the block knobs."""
    tensor_pages = tensor_pages_of(shape)
    page_bytes = page_bytes_of(dtype)
    grid_x, grid_y = grid
    grid_cores = grid_x * grid_y

    banks_per_core = BANKS_PER_CORE
    if num_dram_banks % banks_per_core != 0:
        raise RuntimeError(f"g6_zeros: BANKS_PER_CORE={banks_per_core} must divide {num_dram_banks} DRAM banks")
    num_bank_sets = num_dram_banks // banks_per_core
    if WRITER_CORES_PER_BANK is None:
        max_writer_cores = grid_cores
    else:
        max_writer_cores = min(grid_cores, num_dram_banks * WRITER_CORES_PER_BANK)
    if max_writer_cores < num_bank_sets:
        raise RuntimeError(
            f"g6_zeros: {max_writer_cores} writer cores cannot cover {num_bank_sets} bank sets; "
            "raise WRITER_CORES_PER_BANK or BANKS_PER_CORE"
        )

    # A set's first bank holds the most pages; later banks of the set end one slot earlier at most.
    set_slots = [_ceil_div(max(tensor_pages - s * banks_per_core, 0), num_dram_banks) for s in range(num_bank_sets)]
    # No more cores than slots, so every writer core owns at least one slot.
    num_writer_cores = min(max_writer_cores, sum(set_slots))

    cores = []
    for c in range(num_writer_cores):
        bank_set, rank = c % num_bank_sets, c // num_bank_sets
        group_size = _ceil_div(num_writer_cores - bank_set, num_bank_sets)
        base, extra = divmod(set_slots[bank_set], group_size)
        slots = base + (1 if rank < extra else 0)
        assert slots >= 1, "g6_zeros: a writer core was assigned no slots"
        cores.append(
            CoreWork(
                core=(c % grid_x, c // grid_x),
                bank_set_base=bank_set * banks_per_core,
                first_bank_offset=rank % banks_per_core,
                slot_begin=rank * base + min(rank, extra),
                slots=slots,
                block_slots=slots if BLOCK_SLOTS is None else min(BLOCK_SLOTS, slots),
            )
        )

    max_run_bytes = max(work.block_slots for work in cores) * page_bytes
    return ZerosPlan(
        tensor_pages=tensor_pages,
        page_bytes=page_bytes,
        grid=(grid_x, grid_y),
        num_dram_banks=num_dram_banks,
        banks_per_core=banks_per_core,
        num_writer_cores=num_writer_cores,
        zero_source_bytes=min(ZERO_SOURCE_MAX_BYTES, max_run_bytes),
        cores=tuple(cores),
    )


def _writer_core_ranges(num_writer_cores: int, grid_x: int) -> ttnn.CoreRangeSet:
    """The first num_writer_cores cores in row-major order: full rows plus one partial row."""
    full_rows, partial = divmod(num_writer_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if partial:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(partial - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


# Index of the output buffer address in the writer's common runtime args.
OUTPUT_ADDR_ARG = 0


def create_program_descriptor(output_tensor: ttnn.Tensor, plan: ZerosPlan) -> ttnn.ProgramDescriptor:
    if output_tensor.buffer_num_pages() != plan.tensor_pages:
        raise RuntimeError(
            f"g6_zeros: {output_tensor.buffer_num_pages()} pages allocated, plan has {plan.tensor_pages}"
        )
    if output_tensor.buffer_aligned_page_size() != plan.page_bytes:
        raise RuntimeError(
            f"g6_zeros: aligned page {output_tensor.buffer_aligned_page_size()} B, plan has {plan.page_bytes} B"
        )

    writer_cores = _writer_core_ranges(plan.num_writer_cores, plan.grid[0])

    cb_zero_source = ttnn.CBDescriptor(
        total_size=plan.zero_source_bytes,
        core_ranges=writer_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ZERO_SOURCE,
                data_format=ZERO_SOURCE_CB_FORMAT,
                page_size=plan.zero_source_bytes,
            )
        ],
    )

    ct_args = [CB_ZERO_SOURCE, plan.page_bytes, plan.zero_source_bytes, plan.num_dram_banks, plan.banks_per_core]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    for work in plan.cores:
        x, y = work.core
        rt_args[x][y] = [work.bank_set_base, work.first_bank_offset, work.slot_begin, work.slots, work.block_slots]

    # Order: [OUTPUT_ADDR_ARG] output buffer address, [1] tensor_pages.
    common_rt_args = [output_tensor.buffer_address(), plan.tensor_pages]

    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        core_ranges=writer_cores,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=common_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_zero_source])


def set_output_address(descriptor: ttnn.ProgramDescriptor, output_tensor: ttnn.Tensor) -> None:
    """Patch a cached descriptor for a new output buffer; nothing else in it depends on the call."""
    descriptor.kernels[0].common_runtime_args[OUTPUT_ADDR_ARG] = output_tensor.buffer_address()
