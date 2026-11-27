# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros_like program descriptor.

Each participating core zero-fills an L1 scratch and writes it to the output pages it owns. One writer
kernel per participating core; no reader, no compute kernel. The input tensor is never bound to the
program. Two regimes decide what a core owns:

- `page_split_local_zero`: a contiguous range of the folded page index; one write per page.
- `bank_run_coalesced`: a run of consecutive in-bank pages of one DRAM bank (page ids b, b + B, b + 2B, ...
  with B banks), which are contiguous bytes; written in scratch-sized chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = str(KERNEL_DIR / "g6_zeros_like_writer.cpp")

# Core-assignment knobs.
# Upper bound on participating cores; None = the whole worker grid. With column-major fill, 80 is the
# rectangle (0,0)-(7,9): the whole segment west of the second DRAM column plus that column's nearest two
# worker columns. A few early-starting cores per bank already saturate the bank; the far east columns
# start late and only lengthen the drain (float32 production: 70-90 cores flat, 110 cores ~8% slower).
NUM_CORES_CAP = 80
# Grid fill order of the core set and of the page split. Column-major keeps both production shapes on one
# core rectangle and, under a cap, gives every segment_rows band a core.
ROW_WISE = False

# Regime: "bank_run_coalesced" or "page_split_local_zero". bank_run_coalesced falls back to the page split
# when a bank in use would get no core or every run would be one page (see _bank_run_split).
REGIME = "bank_run_coalesced"
# bank_run_coalesced: which participating core writes which bank's runs.
#   "bank_major": consecutive cores in fill order share a bank.
#   "stagger":    consecutive cores in fill order rotate through the banks.
#   "nearest":    each bank takes the free cores closest to its DRAM-optimal worker for WRITER_NOC.
#   "segment_rows": see _segment_rows; falls back to "bank_major" when a band gets no core or more cores
#                   than its bank has pages.
BANK_ASSIGNMENT = "segment_rows"

# Writer placement: BRISC on the DRAM-write-preferred NoC.
WRITER_PROCESSOR = ttnn.DataMovementProcessor.RISCV_0
WRITER_NOC = ttnn.NOC.NOC_1

# Transfer-size knob: pages of zero scratch per core, clamped to the pages one write may cover (1 in the
# page split, the longest run in bank_run_coalesced). Sizes the CB, the fill and every bank-run chunk.
ZERO_SCRATCH_PAGES = 1

CB_ZERO_PAGE = 0

# Output buffer address: the only per-call value, a common runtime arg patched on every call.
OUTPUT_ADDR_COMMON_ARG = 0


@dataclass(frozen=True)
class CorePlan:
    core: ttnn.CoreCoord
    start_page: int
    num_pages: int
    # Page-id step between the core's pages: 1 in the page split, the bank count in a bank run.
    page_stride: int
    bank: int | None  # the DRAM bank of every page of a bank run; None in the page split


@dataclass(frozen=True)
class Plan:
    regime: str
    num_pages: int
    page_size: int
    zero_scratch_bytes: int
    all_cores: ttnn.CoreRangeSet
    cores: tuple  # tuple[CorePlan, ...]

    @property
    def bank_run_stride(self) -> int:
        """Kernel CT arg: the in-bank page-id stride of a bank run, 0 in the page split."""
        return self.cores[0].page_stride if self.regime == "bank_run_coalesced" else 0


def _num_dram_banks(device) -> int:
    # The interleaved allocator keeps one DRAM bank per DRAM view; the kernel ASSERTs (--dev) that this
    # count is the output accessor's in-bank page stride.
    dram_grid = device.dram_grid_size()
    return dram_grid.x * dram_grid.y


def _participating_cores(device, num_pages):
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_pages, grid.x * grid.y)
    if NUM_CORES_CAP is not None:
        num_cores = min(num_cores, NUM_CORES_CAP)
    return grid, ttnn.num_cores_to_corerangeset(num_cores, grid, ROW_WISE)


def _even_split(total, parts):
    """`parts` counts summing to `total`, the first `total % parts` one larger."""
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def _page_split(core_range_set, num_pages):
    _, all_cores, group_1, group_2, pages_g1, pages_g2 = ttnn.split_work_to_cores(
        core_range_set, num_pages, row_wise=ROW_WISE
    )
    cores = []
    start_page = 0
    for group, pages_per_core in ((group_1, pages_g1), (group_2, pages_g2)):
        if pages_per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=ROW_WISE):
            cores.append(CorePlan(core=core, start_page=start_page, num_pages=pages_per_core, page_stride=1, bank=None))
            start_page += pages_per_core
    return all_cores, cores


def _bank_anchors(device):
    """Per bank, the logical worker the device reports as DRAM-optimal for WRITER_NOC."""
    return ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, WRITER_NOC)


def _segment_rows(device, cores, pages_per_bank):
    """Bank lists for "segment_rows", or None when a bank gets no core or more cores than pages.

    Shaped for a NOC_1 writer, whose packets travel north (-y) and then west (-x). The anchor columns cut
    the grid into segments: a segment runs from one anchor column up to the next, i.e. the worker block
    just east of one DRAM column, which a westward write reaches without crossing another DRAM column. A
    segment's cores go to that column's banks only. Inside a segment the anchor rows cut it into bands:
    a bank takes the rows from its anchor row down to the next anchor row, so its writes climb only
    through its own band and turn west on its own endpoint row. No two banks share a link.
    """
    banks = range(len(pages_per_bank))
    anchors = _bank_anchors(device)
    anchor_cols = sorted({anchors[b].x for b in banks})

    def band_start(value, starts):
        # The nearest start at or before `value`; values before the first start wrap to the last.
        before = [v for v in starts if v <= value]
        return before[-1] if before else starts[-1]

    lists = [[] for _ in banks]
    for col in anchor_cols:
        seg_banks = [b for b in banks if anchors[b].x == col]
        bank_of_row = {anchors[b].y: b for b in seg_banks}
        if len(bank_of_row) != len(seg_banks):
            return None
        anchor_rows = sorted(bank_of_row)
        for core in cores:
            if band_start(core.x, anchor_cols) == col:
                lists[bank_of_row[band_start(core.y, anchor_rows)]].append(core)
    if any(not 0 < len(lists[b]) <= pages_per_bank[b] for b in banks):
        return None
    return lists


def _assign_cores_to_banks(device, cores, pages_per_bank):
    """One non-empty list of cores per bank, at most one core per page of that bank, together exactly `cores`."""
    num_banks = len(pages_per_bank)
    if BANK_ASSIGNMENT == "segment_rows":
        lists = _segment_rows(device, cores, pages_per_bank)
        if lists is not None:
            return lists
    elif BANK_ASSIGNMENT not in ("bank_major", "stagger", "nearest"):
        raise ValueError(f"g6_zeros_like: unknown BANK_ASSIGNMENT {BANK_ASSIGNMENT!r}")

    cores_per_bank = _even_split(len(cores), num_banks)
    if BANK_ASSIGNMENT == "stagger":
        lists = [[] for _ in range(num_banks)]
        order = [b for rnd in range(max(cores_per_bank)) for b in range(num_banks) if rnd < cores_per_bank[b]]
        for core, bank in zip(cores, order):
            lists[bank].append(core)
        return lists
    if BANK_ASSIGNMENT == "nearest":
        anchors = _bank_anchors(device)
        free = list(cores)
        lists = [[] for _ in range(num_banks)]
        for rnd in range(max(cores_per_bank)):
            for bank in range(num_banks):
                if rnd >= cores_per_bank[bank]:
                    continue
                anchor = anchors[bank]
                core = min(free, key=lambda c: abs(c.x - anchor.x) + abs(c.y - anchor.y))
                free.remove(core)
                lists[bank].append(core)
        return lists
    # "bank_major", and the "segment_rows" fallback.
    lists, start = [], 0
    for count in cores_per_bank:
        lists.append(cores[start : start + count])
        start += count
    return lists


def _bank_run_split(device, core_range_set, num_pages, num_banks):
    """Every bank's pages split into in-bank runs, one run per core.

    None (use the page split) if a bank would get no core, or if every run would be one page: a one-page
    run is only a reassignment of the page split's single-page writes, and measured slower (bfloat8_b).
    """
    cores = ttnn.corerange_to_cores(core_range_set, row_wise=ROW_WISE)
    banks_in_use = min(num_pages, num_banks)
    if len(cores) < banks_in_use or len(cores) >= num_pages:
        return None
    pages_per_bank = _even_split(num_pages, num_banks)[:banks_in_use]
    plans = []
    for bank, bank_cores in enumerate(_assign_cores_to_banks(device, cores, pages_per_bank)):
        assert 0 < len(bank_cores) <= pages_per_bank[bank], (bank, len(bank_cores), pages_per_bank[bank])
        in_bank_index = 0
        for core, run_pages in zip(bank_cores, _even_split(pages_per_bank[bank], len(bank_cores))):
            plans.append(
                CorePlan(
                    core=core,
                    start_page=bank + in_bank_index * num_banks,
                    num_pages=run_pages,
                    page_stride=num_banks,
                    bank=bank,
                )
            )
            in_bank_index += run_pages
    assert len(plans) == len(cores), f"g6_zeros_like: {len(cores) - len(plans)} participating cores left idle"
    return plans


def plan(tensor) -> Plan:
    """Work split for a tensor with the output's spec. Every page in [0, P) lands on exactly one core."""
    num_pages = tensor.buffer_num_pages()
    page_size = tensor.buffer_aligned_page_size()
    device = tensor.device()
    _, core_range_set = _participating_cores(device, num_pages)

    cores = None
    if REGIME == "bank_run_coalesced":
        cores = _bank_run_split(device, core_range_set, num_pages, _num_dram_banks(device))
    elif REGIME != "page_split_local_zero":
        raise ValueError(f"g6_zeros_like: unknown REGIME {REGIME!r}")
    if cores is not None:
        regime, all_cores = "bank_run_coalesced", core_range_set
    else:
        regime = "page_split_local_zero"
        all_cores, cores = _page_split(core_range_set, num_pages)

    covered = sorted(
        p for c in cores for p in range(c.start_page, c.start_page + c.num_pages * c.page_stride, c.page_stride)
    )
    assert covered == list(range(num_pages)), f"g6_zeros_like: {regime} does not cover [0, {num_pages}) exactly once"

    scratch_pages = max(
        1, min(ZERO_SCRATCH_PAGES, max(c.num_pages for c in cores) if regime == "bank_run_coalesced" else 1)
    )
    return Plan(
        regime=regime,
        num_pages=num_pages,
        page_size=page_size,
        zero_scratch_bytes=scratch_pages * page_size,
        all_cores=all_cores,
        cores=tuple(cores),
    )


def cache_key(tensor):
    device = tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        "g6_zeros_like",
        tensor.buffer_num_pages(),
        tensor.buffer_aligned_page_size(),
        tensor.dtype,
        grid.x,
        grid.y,
        _num_dram_banks(device),
        NUM_CORES_CAP,
        ROW_WISE,
        REGIME,
        BANK_ASSIGNMENT,
        WRITER_PROCESSOR,
        WRITER_NOC,
        ZERO_SCRATCH_PAGES,
    )


class CachedProgram:
    """A built descriptor whose only per-call input is the output buffer address."""

    def __init__(self, writer: ttnn.KernelDescriptor, cbs: list):
        self._writer = writer
        self._cbs = cbs

    def descriptor_for(self, output_tensor) -> ttnn.ProgramDescriptor:
        # ProgramDescriptor.kernels is exposed by value, so the address is patched on the held
        # KernelDescriptor and the descriptor is assembled from it.
        self._writer.common_runtime_args = [output_tensor.buffer_address()]
        return ttnn.ProgramDescriptor(kernels=[self._writer], semaphores=[], cbs=self._cbs)


def build(output_tensor) -> CachedProgram:
    p = plan(output_tensor)
    # Sizes the CB, the kernel's zero fill and its bank-run chunk: every zero-sourced byte is filled.
    zero_scratch_bytes = p.zero_scratch_bytes

    cb_zero_page = ttnn.CBDescriptor(
        total_size=zero_scratch_bytes,
        core_ranges=p.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ZERO_PAGE,
                data_format=output_tensor.dtype,
                page_size=p.page_size,
            )
        ],
    )

    writer_ct_args = [CB_ZERO_PAGE, p.page_size, zero_scratch_bytes, p.bank_run_stride]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for c in p.cores:
        writer_rt_args[c.core.x][c.core.y] = [c.start_page, c.num_pages]

    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        core_ranges=p.all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.DataMovementConfigDescriptor(processor=WRITER_PROCESSOR, noc=WRITER_NOC),
    )
    return CachedProgram(writer, [cb_zero_page])


_PROGRAM_CACHE: dict = {}


def get_program_descriptor(output_tensor) -> ttnn.ProgramDescriptor:
    key = cache_key(output_tensor)
    program = _PROGRAM_CACHE.get(key)
    if program is None:
        program = build(output_tensor)
        _PROGRAM_CACHE[key] = program
    return program.descriptor_for(output_tensor)
