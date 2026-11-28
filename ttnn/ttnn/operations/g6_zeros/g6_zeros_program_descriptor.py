# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros program descriptor: bank-major zero runs from a core-local zero source.

Regime `bank_run`. Output page p lives in DRAM bank b = p mod N at slot s = p div N, and the
slots of one bank are byte-contiguous. Each writer core owns one contiguous slot run inside one
bank and writes it from a zero region it fills in its own L1. One writer kernel, no reader, no
compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- Blocking knobs: each defined once here; everything else derives from them. ---

# Capacity of the zero source and the largest single NoC write. 16 KiB = one Blackhole
# NOC_MAX_BURST_SIZE packet.
ZERO_SOURCE_BYTES = 16384
# Depth of cb_zero_source. The source never changes, so a second region has nothing to overlap.
ZERO_SOURCE_REGIONS = 1
# Writers per DRAM bank. None selects the Phase 0 rule in phase0_writers_per_bank().
WRITERS_PER_BANK = None

# The L1 the model leaves to the op on every core (BB-05).
MODEL_L1_SHARE_BYTES = 1_347_776

CB_ZERO_SOURCE = 0


@dataclass(frozen=True)
class BankRun:
    """One block: a contiguous slot run inside one bank. One bank per block is the mechanism cap:
    a NoC write targets one DRAM endpoint and slots are contiguous only within a bank."""

    bank: int
    first_page_id: int
    core_slots: int


@dataclass(frozen=True)
class WriterPlan:
    num_pages: int
    num_banks: int
    aligned_page_size: int
    writers_per_bank: int
    zero_source_bytes: int
    zero_source_regions: int
    grid: tuple  # (x, y)
    core_ranges: ttnn.CoreRangeSet
    cores: tuple  # row-wise ordered ttnn.CoreCoord, rank w = index
    blocks: tuple  # per core: tuple[BankRun, ...]


def phase0_writers_per_bank(num_pages: int, num_banks: int, grid_cores: int) -> int:
    return max(1, min(grid_cores // num_banks, -(-num_pages // num_banks)))


def bank_slots(num_pages: int, num_banks: int, bank: int) -> int:
    return num_pages // num_banks + (1 if bank < num_pages % num_banks else 0)


def plan_bank_runs(
    num_pages: int,
    num_banks: int,
    aligned_page_size: int,
    grid,
    *,
    writers_per_bank=WRITERS_PER_BANK,
    zero_source_bytes: int = ZERO_SOURCE_BYTES,
    zero_source_regions: int = ZERO_SOURCE_REGIONS,
) -> WriterPlan:
    grid_cores = grid.x * grid.y
    assert grid_cores >= num_banks, f"g6_zeros: grid of {grid_cores} cores is smaller than {num_banks} DRAM banks"
    dram_alignment = ttnn.get_dram_alignment()
    assert aligned_page_size % dram_alignment == 0, "g6_zeros: page stride must be DRAM-aligned"
    assert zero_source_bytes % dram_alignment == 0, "g6_zeros: zero_source_bytes must be DRAM-aligned"
    assert zero_source_regions * zero_source_bytes <= MODEL_L1_SHARE_BYTES, "g6_zeros: zero source exceeds L1 share"

    if writers_per_bank is None:
        writers_per_bank = phase0_writers_per_bank(num_pages, num_banks, grid_cores)
    assert 1 <= writers_per_bank * num_banks <= grid_cores, "g6_zeros: writers do not fit the grid"

    num_writers = writers_per_bank * num_banks
    core_ranges = ttnn.num_cores_to_corerangeset(num_writers, grid, True)
    cores = tuple(ttnn.corerange_to_cores(core_ranges, None, True))
    assert len(cores) == num_writers

    blocks = []
    for rank in range(num_writers):
        bank = rank % num_banks
        k = rank // num_banks
        slots_b = bank_slots(num_pages, num_banks, bank)
        base, extra = divmod(slots_b, writers_per_bank)
        core_slots = base + (1 if k < extra else 0)
        first_slot = k * base + min(k, extra)
        blocks.append((BankRun(bank, bank + first_slot * num_banks, core_slots),))

    return WriterPlan(
        num_pages=num_pages,
        num_banks=num_banks,
        aligned_page_size=aligned_page_size,
        writers_per_bank=writers_per_bank,
        zero_source_bytes=zero_source_bytes,
        zero_source_regions=zero_source_regions,
        grid=(grid.x, grid.y),
        core_ranges=core_ranges,
        cores=cores,
        blocks=tuple(blocks),
    )


def create_program_descriptor(output_tensor: ttnn.Tensor, plan: WriterPlan) -> ttnn.ProgramDescriptor:
    """Writer-only program. Common runtime arg 0 is the output address, patched on every call."""
    cb_zero_source = ttnn.CBDescriptor(
        total_size=plan.zero_source_regions * plan.zero_source_bytes,
        core_ranges=plan.core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ZERO_SOURCE,
                data_format=output_tensor.dtype,
                page_size=plan.zero_source_bytes,
            )
        ],
    )

    ct_args = [CB_ZERO_SOURCE, plan.aligned_page_size, plan.zero_source_bytes, plan.num_banks]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    for core, core_blocks in zip(plan.cores, plan.blocks):
        args = [len(core_blocks)]
        for run in core_blocks:
            args.extend([run.first_page_id, run.core_slots])
        rt_args[core.x][core.y] = args

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_zeros_writer.cpp"),
        core_ranges=plan.core_ranges,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_zero_source])


def set_output_address(program_descriptor: ttnn.ProgramDescriptor, output_tensor: ttnn.Tensor) -> None:
    # kernels[0] is a reference into the descriptor, so this patches the cached descriptor in place.
    program_descriptor.kernels[0].common_runtime_args[0] = output_tensor.buffer_address()
