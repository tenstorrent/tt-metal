# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_copy program: flat page ranges over the grid, one reader (NCRISC, NoC0) and one writer
(BRISC, NoC1) per core, staged through one CB in coalesced bank runs.

The module-level knobs are read at call time (tests override them with monkeypatch) and are part
of the descriptor cache key.
"""

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# L1 per core the model leaves to the op (BB-05).
L1_BUDGET_BYTES = 1347776
# Cap on cores used; None spreads the pages over the whole compute grid.
MAX_CORES = None
# Cap on pages per block; None lets one block hold a core's whole assignment.
MAX_BLOCK_PAGES = None
# Blocks the staging CB holds on the streamed plan.
STREAM_DEPTH_BLOCKS = 2
# Rotate each core's bank-run walk by its linear index so cores start on different banks.
BANK_STAGGER = False

CB_PAGE_STAGING = 0

READER_KERNEL = "g6_copy_reader.cpp"
WRITER_KERNEL = "g6_copy_writer.cpp"

# Common RT arg slots, identical in both kernels. DRAM-interleaved accessors add no common RT args.
_CRTA_BUFFER_ADDRESS = 0
_CRTA_BLOCK_PAGES = 1


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

    @property
    def cb_pages(self):
        return self.cb_depth_blocks * self.block_pages


def _plan(grid, tensor_pages, page_bytes):
    """Regime selection: the single source for the dispatch and for describe_plan()."""
    if page_bytes > L1_BUDGET_BYTES:
        raise RuntimeError(
            f"g6_copy: a {page_bytes} B page exceeds the {L1_BUDGET_BYTES} B of L1 per core left to the op"
        )

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

    core_pages_max = max(w.num_pages for w in work)
    if core_pages_max * page_bytes <= L1_BUDGET_BYTES and (
        MAX_BLOCK_PAGES is None or core_pages_max <= MAX_BLOCK_PAGES
    ):
        plan, block_pages, cb_depth_blocks = "one_block", core_pages_max, 1
    else:
        block_cap = L1_BUDGET_BYTES // (STREAM_DEPTH_BLOCKS * page_bytes)
        if MAX_BLOCK_PAGES is not None:
            block_cap = min(block_cap, MAX_BLOCK_PAGES)
        block_pages = max(1, block_cap)
        cb_depth_blocks = min(STREAM_DEPTH_BLOCKS, L1_BUDGET_BYTES // (block_pages * page_bytes))
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
    )


def _plan_for(tensor):
    grid = tensor.device().compute_with_storage_grid_size()
    return _plan(grid, tensor.buffer_num_pages(), tensor.buffer_aligned_page_size())


def describe_plan(input_tensor):
    """Report the plan a call on input_tensor selects (BB-08) and the role of every grid core (BB-21)."""
    p = _plan_for(input_tensor)
    roles = {f"({x},{y})": "idle" for y in range(p.grid[1]) for x in range(p.grid[0])}
    for w in p.work:
        roles[
            f"({w.core.x},{w.core.y})"
        ] = f"copy pages [{w.first_page},{w.first_page + w.num_pages}): reader NCRISC/NoC0 + writer BRISC/NoC1"
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
        "l1_bytes_per_core": p.cb_pages * p.page_bytes,
        "bank_stagger": BANK_STAGGER,
        "roles": roles,
    }


def _kernel(source, tensor, plan, config):
    accessor_args = ttnn.TensorAccessorArgs(tensor)
    if len(accessor_args.get_common_runtime_args()) != 0:
        raise RuntimeError("g6_copy: the tensor accessor needs common RT args, which collide with the op's")
    runtime_args = ttnn.RuntimeArgs()
    for w in plan.work:
        runtime_args[w.core.x][w.core.y] = [w.first_page, w.num_pages, w.walk_rotation]
    common_runtime_args = [0] * 2
    common_runtime_args[_CRTA_BUFFER_ADDRESS] = tensor.buffer_address()
    common_runtime_args[_CRTA_BLOCK_PAGES] = plan.block_pages
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / source),
        core_ranges=plan.all_cores,
        compile_time_args=[CB_PAGE_STAGING, *accessor_args.get_compile_time_args()],
        runtime_args=runtime_args,
        common_runtime_args=common_runtime_args,
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
    reader = _kernel(READER_KERNEL, input_tensor, plan, ttnn.ReaderConfigDescriptor())
    writer = _kernel(WRITER_KERNEL, output_tensor, plan, ttnn.WriterConfigDescriptor())
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
        MAX_BLOCK_PAGES,
        STREAM_DEPTH_BLOCKS,
        BANK_STAGGER,
    )
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        plan = _plan(grid, input_tensor.buffer_num_pages(), input_tensor.buffer_aligned_page_size())
        descriptor = _build(input_tensor, output_tensor, plan)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        descriptor.kernels[_READER_INDEX].common_runtime_args[_CRTA_BUFFER_ADDRESS] = input_tensor.buffer_address()
        descriptor.kernels[_WRITER_INDEX].common_runtime_args[_CRTA_BUFFER_ADDRESS] = output_tensor.buffer_address()
    return descriptor
