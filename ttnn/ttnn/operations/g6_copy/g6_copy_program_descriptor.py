# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_copy program: bank-major page ranges, each on a core in the worker row of its DRAM bank's NoC0
endpoint, one reader (NCRISC) and one writer (BRISC) per core, both on NoC0, staged through one CB.

The module-level knobs are read at call time (tests override them with monkeypatch) and are part
of the descriptor cache key.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# L1 per core the model leaves to the op (BB-05).
L1_BUDGET_BYTES = 1347776
# Cap on cores used; None spreads the pages over the whole compute grid.
MAX_CORES = None
# Bytes per block: the overlap knob. A core whose assignment is larger than one block streams it
# through STREAM_DEPTH_BLOCKS slots, so the writer drains block k while the reader fills block k+1;
# a core whose assignment fits one block keeps the single hand-off (no extra blocks for small shapes).
# Expressed in bytes so the block's transfer size carries across page sizes (fp32 tile: 4 pages).
# (16, 12, 128, 128) fp32, 28 pages / core: one_block 67.6 us -> 16 KiB blocks, depth 2, see op_design.md.
# None restores the coarsest-block default (one block per core whenever it fits L1).
BLOCK_BYTES = 16384
# Cap on pages per block (forced tests / sweeps); None leaves the extent to BLOCK_BYTES.
MAX_BLOCK_PAGES = None
# Blocks the staging CB holds on the streamed plan.
STREAM_DEPTH_BLOCKS = 2
# Read-pipeline depth on the streamed plan: blocks of reads kept on the wire, each tagged with its own
# transaction id and retired by a per-id wait instead of a drain-everything read barrier (catalog
# trid_double_issue). 1 is the plain issue -> barrier -> push loop, byte-identical to Refinement 2.
# The staging ring grows by READ_TRIDS - 1 blocks so the writer keeps STREAM_DEPTH_BLOCKS - 1 slots.
READ_TRIDS = 1
# Transaction ids 1..15 are usable (0 means untagged).
_MAX_READ_TRIDS = 15
# Bank-affine work split (design traffic-ranking candidate B, catalog read_coalescing "bank_affinity"):
# the per-core page ranges index a bank-major enumeration of the pages, so a core's pages sit in one
# DRAM bank (two at a boundary) and a block moves as one long bank run instead of up to num_banks
# short ones. Same split, same block schedule; only the page each position names changes.
BANK_AFFINITY = False
# Rotate each core's bank-run walk by its linear index so cores start on different banks. Without it
# every core of a 28-page assignment opens on bank 0 or 4 (16x12x128x128 fp32: 73.9 -> 67.9 us).
BANK_STAGGER = True

# Placement (perf round tiny_placement). The NoC is a unidirectional torus: a DRAM round trip (read
# request + response, write + ack, both on the issuing NoC) costs the full X ring whatever the column,
# plus the full Y ring unless the core sits in the endpoint's row. So the only placement lever is the
# row: a core in its bank's endpoint row skips the Y ring (p150b, 1 page: 905 -> 820 ns per aligned
# kernel, 745 ns with both aligned) and its traffic never leaves that row (no cross-bank contention).
# NoC0 endpoints of all 8 banks land on worker rows, NoC1 endpoints of banks 0 and 4 do not, so both
# kernels use NoC0. Each bank's pages go to k = min(ceil(pages / banks), grid.x) cores: first the free
# cores of the bank's row, then the rows no bank maps to (nearest first), never another bank's row.
KERNEL_NOC = ttnn.NOC.NOC_0

CB_PAGE_STAGING = 0

# Per-stage device zones (kernels/g6_copy_stage_zones.hpp). Off by default: an executed zone costs
# ~51 cycles on the reader -> writer critical path, which was the whole loss to stock ttnn.copy on the
# 1-page cells. G6_COPY_STAGE_ZONES=1 compiles them in for a per-stage breakdown (read at call time).
STAGE_ZONES_ENV = "G6_COPY_STAGE_ZONES"


def _stage_zones():
    return os.environ.get(STAGE_ZONES_ENV, "0") == "1"


READER_KERNEL = "g6_copy_reader.cpp"
WRITER_KERNEL = "g6_copy_writer.cpp"

# Common RT arg slots, identical in both kernels. DRAM-interleaved accessors add no common RT args.
_CRTA_BUFFER_ADDRESS = 0
_CRTA_BLOCK_PAGES = 1
_CRTA_READ_TRIDS = 2
_CRTA_CB_DEPTH_BLOCKS = 3
_CRTA_BANK_MAJOR = 4
_CRTA_TENSOR_PAGES = 5


@dataclass(frozen=True)
class _CoreWork:
    core: ttnn.CoreCoord
    first_page: int
    num_pages: int
    walk_rotation: int


@dataclass(frozen=True)
class _Plan:
    plan: str
    grid: tuple
    all_cores: ttnn.CoreRangeSet
    work: tuple
    tensor_pages: int
    page_bytes: int
    block_pages: int
    cb_depth_blocks: int
    read_trids: int
    bank_major: bool = False

    @property
    def cb_pages(self):
        return self.cb_depth_blocks * self.block_pages


def _split(n, parts):
    q, r = divmod(n, parts)
    out, first = [], 0
    for i in range(parts):
        count = q + (1 if i < r else 0)
        out.append((first, count))
        first += count
    return out


def _bank_row_work(grid, tensor_pages, bank_rows):
    """Bank-major positions: bank b's pages (q + (b < r) of them) over min(k, n_b) cores in row
    bank_rows[b], overflow into the rows no bank maps to (nearest first), then any row."""
    num_banks = len(bank_rows)
    k = min(-(-tensor_pages // num_banks), grid.x)
    q, r = divmod(tensor_pages, num_banks)
    aligned = set(bank_rows)
    used, work, j0 = set(), [], 0

    def free_core(rows):
        for y in rows:
            for x in range(grid.x):
                if (x, y) not in used:
                    used.add((x, y))
                    return ttnn.CoreCoord(x, y)
        return None

    for b in range(num_banks):
        n_b = q + (1 if b < r else 0)
        if n_b == 0:
            continue
        row = bank_rows[b]
        spare = sorted((y for y in range(grid.y) if y not in aligned), key=lambda y: abs(y - row))
        nearest = sorted(range(grid.y), key=lambda y: abs(y - row))
        for first, count in _split(n_b, min(k, n_b)):
            core = free_core([row]) or free_core(spare) or free_core(nearest)
            work.append(_CoreWork(core, j0 + first, count, 0))
        j0 += n_b
    assert j0 == tensor_pages
    return work


def _plan(grid, tensor_pages, page_bytes, num_banks=8, bank_rows=None):
    """Regime selection: the single source for the dispatch and for describe_plan().

    bank_rows[b] is the logical worker row of DRAM bank b's NoC0 endpoint; given (and no MAX_CORES
    cap), the pages are placed bank-major on those rows. Without it the flat row-wise split is used."""
    if page_bytes > L1_BUDGET_BYTES:
        raise RuntimeError(
            f"g6_copy: a {page_bytes} B page exceeds the {L1_BUDGET_BYTES} B of L1 per core left to the op"
        )

    if (
        bank_rows is not None
        and MAX_CORES is None
        and len(bank_rows) == num_banks
        and all(0 <= y < grid.y for y in bank_rows)
    ):
        work = _bank_row_work(grid, tensor_pages, list(bank_rows))
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(w.core, w.core) for w in work])
        return _finish_plan(grid, all_cores, work, tensor_pages, page_bytes, bank_major=True)

    grid_cores = grid.x * grid.y
    pool_cores = grid_cores if MAX_CORES is None else min(MAX_CORES, grid_cores)
    pool = ttnn.num_cores_to_corerangeset(pool_cores, grid, row_wise=True)
    num_cores, all_cores, group_1, group_2, pages_1, pages_2 = ttnn.split_work_to_cores(
        pool, tensor_pages, row_wise=True
    )

    work = []
    first_page = 0
    for group, group_pages in ((group_1, pages_1), (group_2, pages_2)):
        if group_pages == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            rotation = len(work) if BANK_STAGGER else 0
            work.append(_CoreWork(core, first_page, group_pages, rotation))
            first_page += group_pages
    assert len(work) == num_cores and first_page == tensor_pages
    if BANK_AFFINITY and num_cores > 1:
        # Consecutive bank-major ranges share a bank; hand them to cores S = ceil(cores / banks) apart in
        # grid order, so neighbouring cores (which share NoC links) target different banks.
        stride = -(-num_cores // max(1, num_banks))
        order = sorted(range(num_cores), key=lambda r: (r % stride, r // stride))
        work = [
            _CoreWork(work[i].core, work[r].first_page, work[r].num_pages, work[i].walk_rotation)
            for i, r in enumerate(order)
        ]
    return _finish_plan(grid, all_cores, work, tensor_pages, page_bytes, bank_major=BANK_AFFINITY)


def _finish_plan(grid, all_cores, work, tensor_pages, page_bytes, bank_major):
    """Block schedule for a work split (unchanged from the flat split)."""
    core_pages_max = max(w.num_pages for w in work)
    # The block extent the knobs ask for; None means "no cap" (the whole assignment).
    block_target = None
    if BLOCK_BYTES is not None:
        block_target = max(1, BLOCK_BYTES // page_bytes)
    if MAX_BLOCK_PAGES is not None:
        block_target = MAX_BLOCK_PAGES if block_target is None else min(block_target, MAX_BLOCK_PAGES)
    if core_pages_max * page_bytes <= L1_BUDGET_BYTES and (block_target is None or core_pages_max <= block_target):
        plan, block_pages, cb_depth_blocks, read_trids = "one_block", core_pages_max, 1, 1
    else:
        read_trids = max(1, min(READ_TRIDS, _MAX_READ_TRIDS))
        want_depth = STREAM_DEPTH_BLOCKS + read_trids - 1
        block_cap = L1_BUDGET_BYTES // (want_depth * page_bytes)
        if block_target is not None:
            block_cap = min(block_cap, block_target)
        block_pages = max(1, block_cap)
        cb_depth_blocks = min(want_depth, L1_BUDGET_BYTES // (block_pages * page_bytes))
        # The reader holds read_trids slots at its peak; the writer needs at least one more.
        read_trids = max(1, min(read_trids, cb_depth_blocks - 1))
        plan = "streamed"

    return _Plan(
        plan=plan,
        grid=(grid.x, grid.y),
        all_cores=all_cores,
        work=tuple(work),
        tensor_pages=tensor_pages,
        page_bytes=page_bytes,
        block_pages=block_pages,
        cb_depth_blocks=cb_depth_blocks,
        read_trids=read_trids,
        bank_major=bank_major,
    )


def _bank_rows(device):
    return [c.y for c in ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)]


def _num_dram_banks(device):
    dram = device.dram_grid_size()
    return dram.x * dram.y


def _plan_for(tensor):
    grid = tensor.device().compute_with_storage_grid_size()
    device = tensor.device()
    return _plan(
        grid, tensor.buffer_num_pages(), tensor.buffer_aligned_page_size(), _num_dram_banks(device), _bank_rows(device)
    )


def describe_plan(input_tensor):
    """Report the plan a call on input_tensor selects (BB-08) and the role of every grid core (BB-21)."""
    p = _plan_for(input_tensor)
    roles = {f"({x},{y})": "idle" for y in range(p.grid[1]) for x in range(p.grid[0])}
    for w in p.work:
        roles[f"({w.core.x},{w.core.y})"] = (
            f"copy {'bank-major positions' if p.bank_major else 'pages'} [{w.first_page},{w.first_page + w.num_pages}): "
            f"reader NCRISC/{_noc_name(p, True)} + writer BRISC/{_noc_name(p, False)}"
        )
    return {
        "regime": "page_copy",
        "plan": p.plan,
        "grid": p.grid,
        "num_cores": len(p.work),
        "tensor_pages": p.tensor_pages,
        "page_bytes": p.page_bytes,
        "core_pages": [w.num_pages for w in p.work],
        "block_pages": p.block_pages,
        "cb_depth_blocks": p.cb_depth_blocks,
        "cb_pages": p.cb_pages,
        "read_trids": p.read_trids,
        "block_bytes_knob": BLOCK_BYTES,
        "l1_bytes_per_core": p.cb_pages * p.page_bytes,
        "bank_stagger": BANK_STAGGER,
        "bank_affinity": p.bank_major,
        "roles": roles,
    }


def _noc_name(plan, reader):
    if plan.bank_major:
        return "NoC0" if KERNEL_NOC == ttnn.NOC.NOC_0 else "NoC1"
    return "NoC0" if reader else "NoC1"


def _dm_config(plan, reader):
    """Reader NCRISC, writer BRISC; both on KERNEL_NOC on the bank-row plan, else the role defaults."""
    if not plan.bank_major:
        return ttnn.ReaderConfigDescriptor() if reader else ttnn.WriterConfigDescriptor()
    config = ttnn.DataMovementConfigDescriptor()
    config.processor = ttnn.DataMovementProcessor.RISCV_1 if reader else ttnn.DataMovementProcessor.RISCV_0
    config.noc = KERNEL_NOC
    return config


def _kernel(source, tensor, plan, config):
    accessor_args = ttnn.TensorAccessorArgs(tensor)
    if len(accessor_args.get_common_runtime_args()) != 0:
        raise RuntimeError("g6_copy: the tensor accessor needs common RT args, which collide with the op's")
    runtime_args = ttnn.RuntimeArgs()
    for w in plan.work:
        runtime_args[w.core.x][w.core.y] = [w.first_page, w.num_pages, w.walk_rotation]
    common_runtime_args = [0] * 6
    common_runtime_args[_CRTA_BUFFER_ADDRESS] = tensor.buffer_address()
    common_runtime_args[_CRTA_BLOCK_PAGES] = plan.block_pages
    common_runtime_args[_CRTA_READ_TRIDS] = plan.read_trids
    common_runtime_args[_CRTA_CB_DEPTH_BLOCKS] = plan.cb_depth_blocks
    common_runtime_args[_CRTA_BANK_MAJOR] = int(plan.bank_major)
    common_runtime_args[_CRTA_TENSOR_PAGES] = plan.tensor_pages
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / source),
        core_ranges=plan.all_cores,
        compile_time_args=[CB_PAGE_STAGING, *accessor_args.get_compile_time_args()],
        runtime_args=runtime_args,
        common_runtime_args=common_runtime_args,
        defines=[("G6C_STAGE_ZONES", "1")] if _stage_zones() else [],
        config=config,
    )


def _build(input_tensor, output_tensor, plan):
    cb_page_staging = ttnn.CBDescriptor(
        total_size=plan.cb_pages * plan.page_bytes,
        core_ranges=plan.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_PAGE_STAGING,
                data_format=input_tensor.dtype,
                page_size=plan.page_bytes,
            )
        ],
    )
    reader = _kernel(READER_KERNEL, input_tensor, plan, _dm_config(plan, True))
    writer = _kernel(WRITER_KERNEL, output_tensor, plan, _dm_config(plan, False))
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb_page_staging])


_READER_INDEX = 0
_WRITER_INDEX = 1
_DESCRIPTOR_CACHE = {}


def get_program_descriptor(input_tensor, output_tensor):
    """The cached descriptor for this call, with this call's two buffer addresses patched in (BB-07)."""
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    key = (
        device.id(),
        (grid.x, grid.y),
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.buffer_num_pages(),
        input_tensor.buffer_aligned_page_size(),
        L1_BUDGET_BYTES,
        MAX_CORES,
        BLOCK_BYTES,
        MAX_BLOCK_PAGES,
        STREAM_DEPTH_BLOCKS,
        READ_TRIDS,
        BANK_AFFINITY,
        BANK_STAGGER,
        KERNEL_NOC,
        _stage_zones(),
    )
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        plan = _plan(
            grid,
            input_tensor.buffer_num_pages(),
            input_tensor.buffer_aligned_page_size(),
            _num_dram_banks(device),
            _bank_rows(device),
        )
        descriptor = _build(input_tensor, output_tensor, plan)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        descriptor.kernels[_READER_INDEX].common_runtime_args[_CRTA_BUFFER_ADDRESS] = input_tensor.buffer_address()
        descriptor.kernels[_WRITER_INDEX].common_runtime_args[_CRTA_BUFFER_ADDRESS] = output_tensor.buffer_address()
    return descriptor
