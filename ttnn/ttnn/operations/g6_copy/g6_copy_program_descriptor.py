# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_copy program descriptor: plan, descriptor build and host cache.

One CB (`cb_page_staging`), one reader (NCRISC / NoC0) and one writer (BRISC / NoC1) dataflow kernel
over every used core; no compute kernel. Each core copies the flat page range
[core_first_page, core_first_page + core_pages) in blocks of `block_pages` pages, one NoC call per
DRAM bank run.

The module-level knobs are read at call time (never captured at import), so a perf pass or a test
can override them with monkeypatch; every knob is part of the cache key.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- Knobs (single source of truth for every derived quantity) ----------------------------------
# Per-core L1 the model leaves to the op (BB-05).
L1_BUDGET_BYTES = 1347776
# Cap on the number of cores in the pool; None = the full compute grid.
MAX_CORES = None
# Cap on pages per block; None = no cap (one block per core when it fits L1).
MAX_BLOCK_PAGES = None
# Blocks resident in cb_page_staging on the streamed plan (read of block k+1 overlaps write of k).
STREAM_DEPTH_BLOCKS = 2
# Rotate each core's bank-run walk by its linear index so cores do not all start on the same bank.
BANK_STAGGER = False

CB_PAGE_STAGING = 0

_READER_COMMON_ADDR = 0
_WRITER_COMMON_ADDR = 0

_PROGRAM_CACHE = {}


def _grid_of(device):
    return device.compute_with_storage_grid_size()


def _split(grid, tensor_pages):
    """Flat contiguous page ranges over the core pool, row-major, balanced to +-1 page.

    Returns (all_cores, [(core, core_first_page, core_pages), ...]) in row-major order."""
    grid_cores = grid.x * grid.y
    if MAX_CORES is None:
        split = ttnn.split_work_to_cores(grid, tensor_pages, row_wise=True)
    else:
        pool = ttnn.num_cores_to_corerangeset(min(MAX_CORES, grid_cores), grid, row_wise=True)
        split = ttnn.split_work_to_cores(pool, tensor_pages, row_wise=True)
    _, all_cores, core_group_1, core_group_2, pages_g1, pages_g2 = split

    assignment = []
    first_page = 0
    for group, per_core in ((core_group_1, pages_g1), (core_group_2, pages_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, first_page, per_core))
            first_page += per_core
    assert first_page == tensor_pages, f"g6_copy: split covers {first_page} of {tensor_pages} pages"
    return all_cores, assignment


def _plan(input_tensor):
    """Regime selection; the single source for the dispatch and for describe_plan()."""
    grid = _grid_of(input_tensor.device())
    tensor_pages = input_tensor.buffer_num_pages()
    page_bytes = input_tensor.buffer_aligned_page_size()

    if page_bytes > L1_BUDGET_BYTES:
        # sub_page_rows (splitting one page into byte pieces) is not built.
        raise RuntimeError(f"g6_copy: one page of {page_bytes} B exceeds the per-core L1 budget of {L1_BUDGET_BYTES} B")

    all_cores, assignment = _split(grid, tensor_pages)
    core_pages_max = max(pages for _, _, pages in assignment)

    if core_pages_max * page_bytes <= L1_BUDGET_BYTES and (
        MAX_BLOCK_PAGES is None or core_pages_max <= MAX_BLOCK_PAGES
    ):
        plan = "one_block"
        block_pages = core_pages_max
        cb_depth_blocks = 1
    else:
        plan = "streamed"
        block_cap = math.inf if MAX_BLOCK_PAGES is None else MAX_BLOCK_PAGES
        block_pages = max(1, min(block_cap, L1_BUDGET_BYTES // (STREAM_DEPTH_BLOCKS * page_bytes)))
        cb_depth_blocks = min(STREAM_DEPTH_BLOCKS, L1_BUDGET_BYTES // (block_pages * page_bytes))

    walk_rotation = [index if BANK_STAGGER else 0 for index in range(len(assignment))]
    cb_pages = cb_depth_blocks * block_pages

    return {
        "grid": (grid.x, grid.y),
        "all_cores": all_cores,
        "assignment": assignment,
        "walk_rotation": walk_rotation,
        "tensor_pages": tensor_pages,
        "page_bytes": page_bytes,
        "plan": plan,
        "block_pages": block_pages,
        "cb_depth_blocks": cb_depth_blocks,
        "cb_pages": cb_pages,
    }


def describe_plan(input_tensor):
    """Host-only report of the plan a call on `input_tensor` selects (BB-08, BB-21)."""
    p = _plan(input_tensor)
    used = {}
    for core, first_page, pages in p["assignment"]:
        used[
            (core.x, core.y)
        ] = f"copy pages [{first_page},{first_page + pages}): reader NCRISC/NoC0 + writer BRISC/NoC1"
    gx, gy = p["grid"]
    roles = {f"({x},{y})": used.get((x, y), "idle") for y in range(gy) for x in range(gx)}
    return {
        "regime": "page_copy",
        "plan": p["plan"],
        "grid": p["grid"],
        "num_cores": len(p["assignment"]),
        "tensor_pages": p["tensor_pages"],
        "page_bytes": p["page_bytes"],
        "core_pages": [pages for _, _, pages in p["assignment"]],
        "block_pages": p["block_pages"],
        "cb_depth_blocks": p["cb_depth_blocks"],
        "cb_pages": p["cb_pages"],
        "l1_bytes_per_core": p["cb_pages"] * p["page_bytes"],
        "bank_stagger": BANK_STAGGER,
        "roles": roles,
    }


def _build_program_descriptor(input_tensor, output_tensor, p):
    all_cores = p["all_cores"]
    page_bytes = p["page_bytes"]

    cb_page_staging = ttnn.CBDescriptor(
        total_size=p["cb_pages"] * page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_PAGE_STAGING,
                data_format=input_tensor.dtype,
                page_size=page_bytes,
            )
        ],
    )

    reader_ct_args = [CB_PAGE_STAGING]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [CB_PAGE_STAGING]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for (core, first_page, pages), rotation in zip(p["assignment"], p["walk_rotation"]):
        reader_rt_args[core.x][core.y] = [first_page, pages, rotation]
        writer_rt_args[core.x][core.y] = [first_page, pages, rotation]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_copy_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[input_tensor.buffer_address(), p["block_pages"]],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_copy_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address(), p["block_pages"]],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb_page_staging])
    return descriptor, 0, 1


def get_program_descriptor(input_tensor, output_tensor):
    """Cached descriptor with this call's buffer addresses patched in (BB-07)."""
    device = input_tensor.device()
    grid = _grid_of(device)
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
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        entry = _build_program_descriptor(input_tensor, output_tensor, _plan(input_tensor))
        _PROGRAM_CACHE[key] = entry
    descriptor, reader_index, writer_index = entry
    descriptor.kernels[reader_index].common_runtime_args[_READER_COMMON_ADDR] = input_tensor.buffer_address()
    descriptor.kernels[writer_index].common_runtime_args[_WRITER_COMMON_ADDR] = output_tensor.buffer_address()
    return descriptor
