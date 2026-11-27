# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros_like program descriptor.

One regime (`page_split_local_zero`): the output's pages are split, as one folded page index, into
contiguous ranges over the worker grid; each core zero-fills one L1 page and writes it to every page
of its range. One writer kernel per participating core; no reader, no compute kernel. The input
tensor is never bound to the program.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = str(KERNEL_DIR / "g6_zeros_like_writer.cpp")

# Core-assignment knobs.
# Upper bound on participating cores; None = the whole worker grid.
NUM_CORES_CAP = None
# Grid fill order of the page split. Column-major keeps both production shapes on one core rectangle.
ROW_WISE = False

# Writer placement: BRISC on the DRAM-write-preferred NoC.
WRITER_PROCESSOR = ttnn.DataMovementProcessor.RISCV_0
WRITER_NOC = ttnn.NOC.NOC_1

# Buffer-depth knob: pages of zero scratch per core. One page is the source of every write.
ZERO_SCRATCH_PAGES = 1

CB_ZERO_PAGE = 0

# Output buffer address: the only per-call value, a common runtime arg patched on every call.
OUTPUT_ADDR_COMMON_ARG = 0


@dataclass(frozen=True)
class CorePlan:
    core: ttnn.CoreCoord
    start_page: int
    num_pages: int


@dataclass(frozen=True)
class Plan:
    num_pages: int
    page_size: int
    all_cores: ttnn.CoreRangeSet
    cores: tuple  # tuple[CorePlan, ...], in split fill order


def _participating_cores(device, num_pages):
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_pages, grid.x * grid.y)
    if NUM_CORES_CAP is not None:
        num_cores = min(num_cores, NUM_CORES_CAP)
    return grid, ttnn.num_cores_to_corerangeset(num_cores, grid, ROW_WISE)


def plan(tensor) -> Plan:
    """Page split for a tensor with the output's spec. Every page in [0, P) lands on exactly one core."""
    num_pages = tensor.buffer_num_pages()
    page_size = tensor.buffer_aligned_page_size()
    _, core_range_set = _participating_cores(tensor.device(), num_pages)
    _, all_cores, group_1, group_2, pages_g1, pages_g2 = ttnn.split_work_to_cores(
        core_range_set, num_pages, row_wise=ROW_WISE
    )
    cores = []
    start_page = 0
    for group, pages_per_core in ((group_1, pages_g1), (group_2, pages_g2)):
        if pages_per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=ROW_WISE):
            cores.append(CorePlan(core=core, start_page=start_page, num_pages=pages_per_core))
            start_page += pages_per_core
    assert start_page == num_pages, f"g6_zeros_like: split covers {start_page} of {num_pages} pages"
    return Plan(num_pages=num_pages, page_size=page_size, all_cores=all_cores, cores=tuple(cores))


def cache_key(tensor):
    grid = tensor.device().compute_with_storage_grid_size()
    return (
        "g6_zeros_like",
        tensor.buffer_num_pages(),
        tensor.buffer_aligned_page_size(),
        tensor.dtype,
        grid.x,
        grid.y,
        NUM_CORES_CAP,
        ROW_WISE,
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
    # Sizes both the CB and the kernel's zero fill: every zero-sourced byte is filled.
    zero_scratch_bytes = ZERO_SCRATCH_PAGES * p.page_size

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

    writer_ct_args = [CB_ZERO_PAGE, p.page_size, zero_scratch_bytes]
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
