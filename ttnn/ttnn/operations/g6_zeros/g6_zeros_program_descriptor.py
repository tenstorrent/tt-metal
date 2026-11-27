# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_zeros: regime `bank_run_write`.

The output's pages are re-indexed bank-major (bank 0's slots, then bank 1's, ...) and the position
space is split into balanced contiguous ranges dealt to the writer cores in CORE_ORDER. Each writer
zero-fills one L1 source once, then writes every bank run inside its range as one contiguous DRAM
range in packets that all read the same source. One writer kernel on every core of the grid
rectangle; cores without positions return at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = str(KERNEL_DIR / "g6_zeros_writer.cpp")

TILE_DIM = 32
REGIME = "bank_run_write"
ROLE = "zero_writer"

# Defaults are the measured winners on Blackhole (changelog.md); the design's Phase 0 values are
# ZERO_SRC_BYTES_MAX = 16384, WRITER_CORES_PER_BANK = None, CORE_ORDER = "row_major".
#
# Packet-axis knob. The zero fill precedes the first write, so a smaller source starts writing
# sooner; 4 KiB packets still keep a DRAM bank busy. The kernel also clamps to NOC_MAX_BURST_SIZE.
ZERO_SRC_BYTES_MAX = 4096
# Pages of cb_zero_src. The source is filled once and only read afterwards, so a deeper CB buys nothing.
ZERO_SRC_DEPTH = 1
# Core rectangle (x, y) of the kernel and CB, anchored at (0, 0). None = the full compute grid.
WRITER_GRID = None
# Cores per DRAM bank that take positions (writers = this x bank count). None = every core of the
# rectangle. One writer saturates its bank; more writers per bank only add NoC contention.
WRITER_CORES_PER_BANK = 1
# Which cores take positions, in bank-major position order:
#   "row_major"   - the rectangle row-major (x fastest).
#   "bank_affine" - each bank's positions go to the cores nearest that bank's NoC endpoint.
CORE_ORDER = "bank_affine"
CORE_ORDERS = ("row_major", "bank_affine")

WRITER_RISCV = ttnn.DataMovementProcessor.RISCV_0
WRITER_NOC = ttnn.NOC.NOC_1

CB_ZERO_SRC = 0
# cb_zero_src holds raw bytes that never pass the unpacker or packer. A non-block format keeps a page
# of zero_src_bytes legal for every output dtype (16384 B is not a whole number of bfloat8_b tiles).
ZERO_SRC_CB_FORMAT = ttnn.float32

# Common runtime args: [OUTPUT_ADDR_ARG] output buffer address, [TENSOR_PAGES_ARG] P.
OUTPUT_ADDR_ARG = 0
TENSOR_PAGES_ARG = 1


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def tensor_pages_of(shape) -> int:
    d0, d1, d2, d3 = (int(d) for d in shape)
    return d0 * d1 * _ceil_div(d2, TILE_DIM) * _ceil_div(d3, TILE_DIM)


@dataclass(frozen=True)
class Run:
    bank: int
    first_slot: int
    slots: int


@dataclass(frozen=True)
class CoreWork:
    core: tuple  # (x, y) logical worker coordinate
    role: str  # ROLE, or "idle" for a core with no positions
    q_begin: int
    q_count: int
    runs: tuple  # of Run; what the kernel derives from (q_begin, q_count) on its own


@dataclass(frozen=True)
class ZerosPlan:
    regime: str
    shape: tuple
    dtype: object
    grid: tuple  # (x, y) of the kernel's core rectangle
    core_order: str
    num_dram_banks: int  # reporting only: the kernel takes NUM_DRAM_BANKS
    tensor_pages: int
    aligned_page_bytes: int
    cores_used: int
    zero_src_bytes: int
    zero_src_depth: int
    packet_bytes: int  # before the kernel's NOC_MAX_BURST_SIZE clamp
    writer_riscv: str
    writer_noc: str
    cores: tuple  # of CoreWork: every core of the rectangle, writers in position order, then idle cores


@dataclass(frozen=True)
class Geometry:
    grid: tuple  # (x, y) kernel rectangle
    num_dram_banks: int
    bank_anchors: tuple  # per DRAM bank: the logical worker (x, y) nearest its endpoint on WRITER_NOC


def device_geometry(device) -> Geometry:
    """Device facts the plan depends on; grid is the WRITER_GRID rectangle inside the compute grid."""
    full = device.compute_with_storage_grid_size()
    grid_x, grid_y = (int(full.x), int(full.y)) if WRITER_GRID is None else WRITER_GRID
    if not (1 <= grid_x <= full.x and 1 <= grid_y <= full.y):
        raise RuntimeError(f"g6_zeros: WRITER_GRID {WRITER_GRID} outside the compute grid {full}")
    dram = device.dram_grid_size()
    num_dram_banks = int(dram.x) * int(dram.y)
    # One program runs on every chip of the mesh, so the placement follows the first chip's banks.
    anchors = device.get_optimal_dram_bank_to_logical_worker_assignment(WRITER_NOC, ttnn.MeshCoordinate(0, 0))
    if isinstance(anchors, dict):
        anchors = [anchors[bank] for bank in sorted(anchors)]
    bank_anchors = tuple((int(core.x), int(core.y)) for core in anchors)
    if len(bank_anchors) != num_dram_banks:
        raise RuntimeError(f"g6_zeros: {len(bank_anchors)} bank anchors for {num_dram_banks} DRAM banks")
    return Geometry(grid=(grid_x, grid_y), num_dram_banks=num_dram_banks, bank_anchors=bank_anchors)


def plan_knobs() -> tuple:
    """Every module knob the plan reads; part of the descriptor cache key."""
    return (ZERO_SRC_BYTES_MAX, ZERO_SRC_DEPTH, WRITER_CORES_PER_BANK, CORE_ORDER)


def bank_runs(q_begin: int, q_count: int, tensor_pages: int, num_dram_banks: int) -> tuple:
    """Bank runs covering bank-major positions [q_begin, q_begin + q_count); mirrors the kernel."""
    slots_per_bank = _ceil_div(tensor_pages, num_dram_banks)
    full_banks = tensor_pages - num_dram_banks * (slots_per_bank - 1)
    full_span = full_banks * slots_per_bank
    runs = []
    q, q_end = q_begin, q_begin + q_count
    while q < q_end:
        if q < full_span:
            bank, slot, bank_slots = q // slots_per_bank, q % slots_per_bank, slots_per_bank
        else:
            bank_offset, slot = divmod(q - full_span, slots_per_bank - 1)
            bank, bank_slots = full_banks + bank_offset, slots_per_bank - 1
        run_slots = min(q_end - q, bank_slots - slot)
        runs.append(Run(bank=bank, first_slot=slot, slots=run_slots))
        q += run_slots
    return tuple(runs)


def _bank_of(q: int, tensor_pages: int, num_dram_banks: int) -> int:
    return bank_runs(q, 1, tensor_pages, num_dram_banks)[0].bank


def _writer_cores(geometry: Geometry, counts: list, tensor_pages: int) -> list:
    """The cores that take the position ranges `counts` (in position order), per CORE_ORDER."""
    grid_x, grid_y = geometry.grid
    rectangle = [(index % grid_x, index // grid_x) for index in range(grid_x * grid_y)]
    if CORE_ORDER == "row_major":
        return rectangle[: len(counts)]

    # bank_affine: a range belongs to the bank holding its middle position. Each bank takes as many
    # cores as it has ranges, nearest-first to its anchor; ties go to the lower bank, then row-major.
    range_banks, q_begin = [], 0
    for count in counts:
        range_banks.append(_bank_of(q_begin + (count - 1) // 2, tensor_pages, geometry.num_dram_banks))
        q_begin += count
    quota = [range_banks.count(bank) for bank in range(geometry.num_dram_banks)]

    def distance(core, bank):
        anchor_x, anchor_y = geometry.bank_anchors[bank]
        return abs(core[0] - anchor_x) + abs(core[1] - anchor_y)

    pairs = sorted(
        (distance(core, bank), bank, core[1], core[0])
        for core in rectangle
        for bank in range(geometry.num_dram_banks)
        if quota[bank]
    )
    taken, cores_of_bank = set(), [[] for _ in range(geometry.num_dram_banks)]
    for _, bank, y, x in pairs:
        if (x, y) in taken or len(cores_of_bank[bank]) == quota[bank]:
            continue
        taken.add((x, y))
        cores_of_bank[bank].append((x, y))
    return [cores_of_bank[bank].pop(0) for bank in range_banks]


def make_plan(shape, dtype, geometry: Geometry) -> ZerosPlan:
    """Balanced contiguous bank-major position ranges, dealt to the writer cores in CORE_ORDER."""
    if CORE_ORDER not in CORE_ORDERS:
        raise RuntimeError(f"g6_zeros: CORE_ORDER {CORE_ORDER!r} not in {CORE_ORDERS}")
    shape = tuple(int(d) for d in shape)
    tensor_pages = tensor_pages_of(shape)
    aligned_page_bytes = ttnn.tile_size(dtype)
    if aligned_page_bytes % 64:
        raise RuntimeError(f"g6_zeros: page of {aligned_page_bytes} B is not 64 B aligned")

    grid_x, grid_y = geometry.grid
    num_cores = grid_x * grid_y
    if WRITER_CORES_PER_BANK is not None:
        num_cores = min(num_cores, WRITER_CORES_PER_BANK * geometry.num_dram_banks)
    if num_cores < 1:
        raise RuntimeError(f"g6_zeros: WRITER_CORES_PER_BANK {WRITER_CORES_PER_BANK} leaves no writer")
    cores_used, _, group_1, group_2, positions_1, positions_2 = ttnn.split_work_to_cores(
        ttnn.CoreCoord(num_cores, 1), tensor_pages, row_wise=True
    )
    group_1_cores, group_2_cores = group_1.num_cores(), group_2.num_cores()
    assert group_1_cores + group_2_cores == cores_used
    assert group_1_cores * positions_1 + group_2_cores * positions_2 == tensor_pages
    counts = [positions_1] * group_1_cores + [positions_2] * group_2_cores

    zero_src_bytes = min(ZERO_SRC_BYTES_MAX, positions_1 * aligned_page_bytes)

    writers = _writer_cores(geometry, counts, tensor_pages)
    cores, q_begin = [], 0
    for core, q_count in zip(writers, counts):
        runs = bank_runs(q_begin, q_count, tensor_pages, geometry.num_dram_banks)
        cores.append(CoreWork(core=core, role=ROLE, q_begin=q_begin, q_count=q_count, runs=runs))
        q_begin += q_count
    assert q_begin == tensor_pages
    busy = set(writers)
    for index in range(grid_x * grid_y):
        core = (index % grid_x, index // grid_x)
        if core not in busy:
            cores.append(CoreWork(core=core, role="idle", q_begin=tensor_pages, q_count=0, runs=()))

    return ZerosPlan(
        regime=REGIME,
        shape=shape,
        dtype=dtype,
        grid=(grid_x, grid_y),
        core_order=CORE_ORDER,
        num_dram_banks=geometry.num_dram_banks,
        tensor_pages=tensor_pages,
        aligned_page_bytes=aligned_page_bytes,
        cores_used=cores_used,
        zero_src_bytes=zero_src_bytes,
        zero_src_depth=ZERO_SRC_DEPTH,
        packet_bytes=zero_src_bytes,
        writer_riscv=str(WRITER_RISCV),
        writer_noc=str(WRITER_NOC),
        cores=tuple(cores),
    )


def create_program_descriptor(output_tensor: ttnn.Tensor, plan: ZerosPlan) -> ttnn.ProgramDescriptor:
    if output_tensor.buffer_aligned_page_size() != plan.aligned_page_bytes:
        raise RuntimeError(
            f"g6_zeros: output page {output_tensor.buffer_aligned_page_size()} B, plan page {plan.aligned_page_bytes} B"
        )
    if output_tensor.buffer_num_pages() != plan.tensor_pages:
        raise RuntimeError(f"g6_zeros: output has {output_tensor.buffer_num_pages()} pages, plan {plan.tensor_pages}")

    grid_x, grid_y = plan.grid
    # One rectangle (BB-17); cores without positions get an empty range and return at once.
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])

    cb_zero_src = ttnn.CBDescriptor(
        total_size=plan.zero_src_depth * plan.zero_src_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ZERO_SRC,
                data_format=ZERO_SRC_CB_FORMAT,
                page_size=plan.zero_src_bytes,
            )
        ],
    )

    ct_args = [CB_ZERO_SRC, plan.zero_src_bytes]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    for work in plan.cores:
        x, y = work.core
        rt_args[x][y] = [work.q_begin, work.q_count]

    common_rt_args = [0] * 2
    common_rt_args[OUTPUT_ADDR_ARG] = output_tensor.buffer_address()
    common_rt_args[TENSOR_PAGES_ARG] = plan.tensor_pages

    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        core_ranges=all_cores,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=common_rt_args,
        config=ttnn.DataMovementConfigDescriptor(processor=WRITER_RISCV, noc=WRITER_NOC),
    )
    return ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_zero_src])


def set_output_address(descriptor: ttnn.ProgramDescriptor, output_tensor: ttnn.Tensor) -> None:
    """Patch a cached descriptor for a new output buffer; nothing else in it depends on the call."""
    descriptor.kernels[0].common_runtime_args[OUTPUT_ADDR_ARG] = output_tensor.buffer_address()
