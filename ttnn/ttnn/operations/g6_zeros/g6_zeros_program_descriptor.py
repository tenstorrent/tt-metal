# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros — plan + ProgramDescriptor (regime R1: bank-affine coalesced zero runs).

Every knob of the Blocking Model is a single-source parameter here:
  * ``cores_per_bank`` — how many cores split one DRAM bank's bank-rows (grid-sync lamp)
  * ``burst_pages``    — pages per NoC write; sizes the zero source (burst lamp)
  * ``MAX_WRITE_BURST_BYTES`` — the cap ``burst_pages`` is derived from
  * ``placement``      — which grid core serves which (bank, slot): ``"row_major"`` (ordinal k ->
    k-th core in row-major order) or ``"bank_near"`` (each bank's cores are the free cores nearest
    that bank's NoC-optimal worker, ``get_optimal_dram_bank_to_logical_worker_assignment``)
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
    placement: str
    cores: tuple  # cores[k] = (x, y) of core ordinal k; the single source of the bank -> core map

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
            self.placement,
            self.cores,
        )

    def core_coord(self, k: int) -> ttnn.CoreCoord:
        """Core of ordinal k (serves bank k % B, slot k // B)."""
        x, y = self.cores[k]
        return ttnn.CoreCoord(x, y)

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
        """Used cores, merged into maximal horizontal runs per grid row (few rectangular ranges)."""
        rows = {}
        for x, y in self.cores:
            rows.setdefault(y, []).append(x)
        ranges = []
        for y in sorted(rows):
            xs = sorted(rows[y])
            lo = prev = xs[0]
            for x in xs[1:] + [None]:
                if x is not None and x == prev + 1:
                    prev = x
                    continue
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(lo, y), ttnn.CoreCoord(prev, y)))
                if x is not None:
                    lo = prev = x
        return ttnn.CoreRangeSet(ranges)


def num_dram_banks(device) -> int:
    return ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks


PLACEMENTS = ("row_major", "bank_near")
DEFAULT_PLACEMENT = "bank_near"
DEFAULT_CORES_PER_BANK = 1


def _place_cores(device, gx: int, gy: int, B: int, cores_per_bank: int, placement: str) -> tuple:
    """cores[k] for ordinal k = slot * B + bank. One source of truth for the bank -> core map."""
    n = B * cores_per_bank
    if placement == "row_major":
        return tuple((k % gx, k // gx) for k in range(n))
    assert placement == "bank_near", f"g6_zeros: unknown placement {placement!r}"
    # The writer runs on BRISC / NoC1 (WriterConfigDescriptor), so anchor on the NoC1-optimal workers.
    anchors = [(c.x, c.y) for c in ttnn.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_1)]
    assert len(anchors) >= B, (len(anchors), B)
    free = {(x, y) for x in range(gx) for y in range(gy)}
    cores = []
    for s in range(cores_per_bank):
        for b in range(B):
            ax, ay = anchors[b]
            # Nearest free core; prefer the anchor's own row (short, link-disjoint write path to the bank).
            c = min(free, key=lambda p: (abs(p[1] - ay) * 2 + abs(p[0] - ax), p[1] != ay, p))
            free.remove(c)
            cores.append(c)
    return tuple(cores)


def make_plan(
    num_pages: int,
    aligned_page_size: int,
    device,
    *,
    cores_per_bank=None,
    burst_pages=None,
    placement=None,
) -> G6ZerosPlan:
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    B = num_dram_banks(device)
    G = gx * gy
    if cores_per_bank is None:
        cores_per_bank = DEFAULT_CORES_PER_BANK
    cores_per_bank = max(1, min(cores_per_bank, G // B, -(-num_pages // B)))
    if burst_pages is None:
        burst_pages = max(1, MAX_WRITE_BURST_BYTES // aligned_page_size)
    if placement is None:
        placement = DEFAULT_PLACEMENT
    cores = _place_cores(device, gx, gy, B, cores_per_bank, placement)
    plan = G6ZerosPlan(num_pages, aligned_page_size, B, gx, gy, cores_per_bank, burst_pages, placement, cores)
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
