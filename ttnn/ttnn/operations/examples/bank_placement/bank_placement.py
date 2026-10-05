# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""NoC-placement example: putting each DRAM bank's reader on the core NEAREST that bank.

An interleaved DRAM tensor puts page p in bank `p % num_banks`. When a core's work
is "every page of ONE bank" (pages b, b + num_banks, b + 2*num_banks, ...), that core
has a home bank, and WHICH core does it decides how far every byte travels over the
NoC and which routes the cores share. The device can report, per bank, the worker
core it considers best placed for that bank.

The op is a DRAM -> DRAM copy (interleaved in, interleaved out, same page indices),
one core per bank, so both the read stream (NoC0) and the write stream (NoC1) go to
that core's bank. Three placements of the SAME work:

    placement="row_major"     : bank b's work on core b of a row-major grid fill.
    placement="bank_near"     : bank b's work on the device's preferred core for bank b.
    placement="bank_shuffled" : the SAME cores as bank_near, but each one serves the bank
                                half-way round the list -- separates "near my bank"
                                from "spread over the grid".

and two access patterns:

    pattern="affine" : core b copies bank b's pages (stride = num_banks) -- a home bank.
    pattern="spread" : core k copies a contiguous run of pages (stride = 1) -- every core
                       touches every bank, so there is no home bank (control).

Same kernels, same transactions, same counts; only the core <-> work mapping moves.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB = 0
CB_DEPTH = 2  # blocks; held constant

# Baseline first.
VARIANTS = ("row_major", "bank_near", "bank_shuffled")
PATTERNS = ("affine", "spread")


def num_banks(device):
    return len(ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, 0))


def placement_cores(device, placement):
    """One core per DRAM bank; list index == bank id."""
    near = list(ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, 0))
    nb = len(near)
    if placement == "bank_near":
        return near
    if placement == "bank_shuffled":
        return [near[(b + nb // 2) % nb] for b in range(nb)]
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(nb)]


def work_ranges(pattern, num_pages, nb):
    """(first, stride, count) per worker index."""
    per = num_pages // nb
    if pattern == "affine":
        return [(b, nb, per) for b in range(nb)]
    return [(k * per, 1, per) for k in range(nb)]


def validate(input_tensor, nb):
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or len(input_tensor.shape) != 2:
        raise ValueError("bank_placement example: input must be a 2D ROW_MAJOR tensor")
    if input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("bank_placement example: input must be interleaved")
    if input_tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("bank_placement example: input must be in DRAM")
    h = input_tensor.shape[0]
    if h % nb:
        raise ValueError(f"bank_placement example: rows (pages) must be a multiple of {nb} DRAM banks, got {h}")


def create_program_descriptor(input_tensor, output_tensor, *, variant, pattern, block, kernel_iters=1):
    if variant not in VARIANTS:
        raise ValueError(f"bank_placement example: variant must be one of {VARIANTS}, got {variant!r}")
    if pattern not in PATTERNS:
        raise ValueError(f"bank_placement example: pattern must be one of {PATTERNS}, got {pattern!r}")
    device = input_tensor.device()
    cores = placement_cores(device, variant)
    nb = len(cores)
    page_bytes = input_tensor.buffer_aligned_page_size()
    ranges = work_ranges(pattern, input_tensor.shape[0], nb)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    cb = ttnn.CBDescriptor(
        total_size=CB_DEPTH * block * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )
    reader_ct = [page_bytes, block, kernel_iters] + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [page_bytes, block, kernel_iters] + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for core, (first, stride, count) in zip(cores, ranges):
        reader_rt[core.x][core.y] = [in_addr, first, stride, count]
        writer_rt[core.x][core.y] = [out_addr, first, stride, count]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bp_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bp_writer.cpp"),
            core_ranges=core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb])


def bank_placement(
    input_tensor: ttnn.Tensor,
    *,
    variant: str = "bank_near",
    pattern: str = "affine",
    block: int = 8,
    kernel_iters: int = 1,
) -> ttnn.Tensor:
    """Copy an interleaved DRAM ROW_MAJOR tensor to a new one, one core per DRAM bank.

    Args:
        variant: "row_major" | "bank_near" | "bank_shuffled" -- which core serves which bank.
        pattern: "affine" (each core copies one bank's pages) | "spread" (contiguous runs).
        block: pages per NoC barrier / CB block.
        kernel_iters: in-kernel repeat. 1 = per-launch latency.
    """
    if block < 1 or kernel_iters < 1:
        raise ValueError("bank_placement example: block and kernel_iters must be >= 1")
    device = input_tensor.device()
    validate(input_tensor, num_banks(device))
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = create_program_descriptor(
        input_tensor, output_tensor, variant=variant, pattern=pattern, block=block, kernel_iters=kernel_iters
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
