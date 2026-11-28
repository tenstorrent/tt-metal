# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros — plan + ProgramDescriptor (regime R1: bank-affine coalesced zero runs).

Every knob of the Blocking Model is a single-source parameter here:
  * ``cores_per_bank`` — how many cores split one DRAM bank's bank-rows (grid-sync lamp)
  * ``burst_pages``    — pages per NoC write; sizes the zero source (burst lamp)
  * ``MAX_WRITE_BURST_BYTES`` — the cap ``burst_pages`` is derived from
Per-core ``(first_page, run_pages)`` and the core ranges are derived from them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Blackhole NOC_MAX_BURST_SIZE: the widest single NoC write one zero source of <= 16 KiB feeds.
MAX_WRITE_BURST_BYTES = 16384

CB_ZERO_SOURCE = 0  # L1 scratch holding burst_pages * aps zero bytes; source of every NoC write


@dataclass(frozen=True)
class G6ZerosPlan:
    num_pages: int  # N
    aligned_page_size: int  # aps
    num_banks: int  # B
    grid_x: int
    grid_y: int
    cores_per_bank: int
    burst_pages: int

    @property
    def num_cores(self) -> int:
        return self.num_banks * self.cores_per_bank

    @property
    def zero_source_bytes(self) -> int:
        return self.burst_pages * self.aligned_page_size

    def key(self):
        return (
            self.num_pages,
            self.aligned_page_size,
            self.num_banks,
            self.grid_x,
            self.grid_y,
            self.cores_per_bank,
            self.burst_pages,
        )

    def core_coord(self, k: int) -> ttnn.CoreCoord:
        """Core ordinal k in row-major grid order."""
        return ttnn.CoreCoord(k % self.grid_x, k // self.grid_x)

    def core_runs(self):
        """Yield (ordinal, first_page, run_pages) for every used core."""
        B, cpb, N = self.num_banks, self.cores_per_bank, self.num_pages
        for k in range(self.num_cores):
            b, s = k % B, k // B
            n_b = (N - b + B - 1) // B if b < N else 0
            q, rem = divmod(n_b, cpb)
            run_pages = q + (1 if s < rem else 0)
            row_start = s * q + min(s, rem)
            yield k, row_start * B + b, run_pages

    def core_range_set(self) -> ttnn.CoreRangeSet:
        """First num_cores cores in row-major order: full rows + at most one partial row."""
        n, gx = self.num_cores, self.grid_x
        full_rows, tail = divmod(n, gx)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
        if tail:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(tail - 1, full_rows)))
        return ttnn.CoreRangeSet(ranges)


def num_dram_banks(device) -> int:
    return ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks


def make_plan(num_pages: int, aligned_page_size: int, device, *, cores_per_bank=None, burst_pages=None) -> G6ZerosPlan:
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    B = num_dram_banks(device)
    G = gx * gy
    if cores_per_bank is None:
        cores_per_bank = min(G // B, -(-num_pages // B))
    cores_per_bank = max(1, min(cores_per_bank, G // B))
    if burst_pages is None:
        burst_pages = max(1, MAX_WRITE_BURST_BYTES // aligned_page_size)
    plan = G6ZerosPlan(num_pages, aligned_page_size, B, gx, gy, cores_per_bank, burst_pages)
    covered = sum(r for _, _, r in plan.core_runs())
    assert covered == num_pages, f"g6_zeros plan covers {covered} of {num_pages} pages"
    return plan


def _stable_hash(key) -> int:
    digest = hashlib.blake2b(repr(("g6_zeros", key)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def create_program_descriptor(output_tensor: ttnn.Tensor, plan: G6ZerosPlan) -> ttnn.ProgramDescriptor:
    core_grid = plan.core_range_set()

    cb_zero_source = ttnn.CBDescriptor(
        total_size=plan.zero_source_bytes,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ZERO_SOURCE,
                data_format=output_tensor.dtype,
                page_size=plan.zero_source_bytes,
            )
        ],
    )

    ct_args = [plan.num_banks, plan.burst_pages, plan.aligned_page_size, CB_ZERO_SOURCE]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    for k, first_page, run_pages in plan.core_runs():
        c = plan.core_coord(k)
        rt_args[c.x][c.y] = [first_page, run_pages]

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_zeros_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    desc = ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_zero_source])
    desc.custom_program_hash = _stable_hash((str(output_tensor.dtype),) + plan.key())
    return desc
