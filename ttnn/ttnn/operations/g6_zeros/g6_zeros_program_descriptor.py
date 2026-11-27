# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros program descriptor: bank-run zero write.

Work split (bank-affine): worker core c (row-major over the worker grid) owns DRAM bank c mod B and
a contiguous share of that bank's rows. Within one bank consecutive rows are consecutive addresses,
so each core's share is one contiguous DRAM run, written from one Z-byte zeroed L1 source.

The kernel takes a bank-major page range [range_start, range_start + range_count): bank-major
position q = prefix(b) + row maps to page id row * B + b. Any placement of ranges on cores is a
host-only change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = KERNEL_DIR / "g6_zeros_writer.cpp"

CB_ZERO_SOURCE = 0

# NOC_MAX_BURST_SIZE (tt-1xx noc_parameters.h); the kernel static_asserts Z against the real macro.
_NOC_MAX_BURST_SIZE = {
    ttnn.Arch.BLACKHOLE: 256 * 64,
    ttnn.Arch.WORMHOLE_B0: 256 * 32,
}


@dataclass(frozen=True)
class Knobs:
    # Worker cores W, enumerated row-major from (0, 0); None = the whole worker grid.
    # Clamped to [B, grid cores]: every bank needs at least one core.
    num_worker_cores: int | None = None
    # Upper bound on the transfer quantum Z; None = NOC_MAX_BURST_SIZE.
    zero_chunk_bytes_cap: int | None = None


DEFAULT_KNOBS = Knobs()


@dataclass(frozen=True)
class CoreAssignment:
    x: int
    y: int
    role: str  # "zero_writer" or "idle"
    bank: int
    start_row: int
    core_bank_rows: int
    range_start: int  # bank-major page position of the core's first row


@dataclass(frozen=True)
class Plan:
    regime: str
    num_pages: int
    page_bytes: int
    num_banks: int
    num_worker_cores: int
    zero_chunk_bytes: int
    cores: tuple[CoreAssignment, ...]


def _num_banks(device):
    dram = device.dram_grid_size()
    return dram.x * dram.y


def _plan(num_pages, page_bytes, device, knobs):
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    num_banks = _num_banks(device)
    assert num_banks <= grid_cores, f"g6_zeros: {num_banks} DRAM banks but only {grid_cores} worker cores"

    requested = grid_cores if knobs.num_worker_cores is None else knobs.num_worker_cores
    num_workers = max(num_banks, min(requested, grid_cores))

    rows_per_bank, banks_with_extra_row = divmod(num_pages, num_banks)
    cores = []
    for c in range(num_workers):
        bank = c % num_banks
        slot = c // num_banks
        cores_of_bank = (num_workers - 1 - bank) // num_banks + 1
        rows_in_bank = rows_per_bank + (1 if bank < banks_with_extra_row else 0)
        base, rem = divmod(rows_in_bank, cores_of_bank)
        start_row = slot * base + min(slot, rem)
        core_bank_rows = base + (1 if slot < rem else 0)
        bank_prefix = bank * rows_per_bank + min(bank, banks_with_extra_row)
        cores.append(
            CoreAssignment(
                x=c % grid.x,
                y=c // grid.x,
                role="zero_writer" if core_bank_rows > 0 else "idle",
                bank=bank,
                start_row=start_row,
                core_bank_rows=core_bank_rows,
                range_start=bank_prefix + start_row,
            )
        )

    # The ranges must tile [0, num_pages) exactly once: a gap is a page never written.
    covered = 0
    for core in sorted((c for c in cores if c.core_bank_rows > 0), key=lambda c: c.range_start):
        assert core.range_start == covered, f"g6_zeros: bank-major ranges leave a gap or overlap at {covered}"
        covered += core.core_bank_rows
    assert covered == num_pages, f"g6_zeros: ranges cover {covered} of {num_pages} pages"

    largest_run_bytes = max(c.core_bank_rows for c in cores) * page_bytes
    zero_chunk_bytes = min(_NOC_MAX_BURST_SIZE[device.arch()], largest_run_bytes)
    if knobs.zero_chunk_bytes_cap is not None:
        zero_chunk_bytes = min(zero_chunk_bytes, knobs.zero_chunk_bytes_cap)
    # Chunk starts are run start + k * Z; a DRAM-aligned Z keeps every write start aligned.
    assert zero_chunk_bytes > 0 and zero_chunk_bytes % ttnn.get_dram_alignment() == 0, zero_chunk_bytes

    return Plan(
        regime="bank_run",
        num_pages=num_pages,
        page_bytes=page_bytes,
        num_banks=num_banks,
        num_worker_cores=num_workers,
        zero_chunk_bytes=zero_chunk_bytes,
        cores=tuple(cores),
    )


def _worker_core_ranges(num_workers, grid):
    """The first num_workers cores of the grid, row-major: at most a full-row rectangle plus one row."""
    full_rows, partial = divmod(num_workers, grid.x)
    ranges = []
    if full_rows > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
    if partial > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(partial - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _page_bytes(output):
    page_bytes = output.buffer_aligned_page_size()
    # Bank runs are contiguous only when the interleave stride is the page size itself.
    assert output.buffer_page_size() == page_bytes, (output.buffer_page_size(), page_bytes)
    return page_bytes


def create_program_descriptor(output, device, plan):
    grid = device.compute_with_storage_grid_size()
    core_ranges = _worker_core_ranges(plan.num_worker_cores, grid)

    # Raw L1 bytes, never unpacked or packed; declared in a non-block format so Z need not be a
    # whole number of block-float tiles.
    cb_zero_source = ttnn.CBDescriptor(
        total_size=plan.zero_chunk_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ZERO_SOURCE,
                data_format=ttnn.bfloat16,
                page_size=plan.zero_chunk_bytes,
            )
        ],
    )

    writer_ct_args = [CB_ZERO_SOURCE, plan.zero_chunk_bytes, plan.page_bytes, plan.num_banks]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for core in plan.cores:
        writer_rt_args[core.x][core.y] = [core.range_start, core.core_bank_rows]

    writer = ttnn.KernelDescriptor(
        kernel_source=str(WRITER_KERNEL),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output.buffer_address(), plan.num_pages],
        config=ttnn.WriterConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_zero_source])


# Built once per key; a hit only rewrites the output address (common RT arg 0).
_PROGRAMS = {}


def _cache_key(output, device, knobs):
    grid = device.compute_with_storage_grid_size()
    memory_config = output.memory_config()
    return (
        device.id(),
        tuple(device.shape),
        (grid.x, grid.y),
        output.buffer_num_pages(),
        output.dtype,
        output.layout,
        memory_config.memory_layout,
        memory_config.buffer_type,
        knobs,
    )


def get_program(output, device, knobs=DEFAULT_KNOBS):
    key = _cache_key(output, device, knobs)
    entry = _PROGRAMS.get(key)
    if entry is None:
        plan = _plan(output.buffer_num_pages(), _page_bytes(output), device, knobs)
        entry = (create_program_descriptor(output, device, plan), plan)
        _PROGRAMS[key] = entry
    descriptor, _ = entry
    descriptor.kernels[0].common_runtime_args[0] = output.buffer_address()
    return descriptor


def plan_for(output, device, knobs=DEFAULT_KNOBS):
    return _plan(output.buffer_num_pages(), _page_bytes(output), device, knobs)
