# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Page-index walk order: the temporal ORDER a single reader walks its source pages.

Interleaved DRAM round-robins consecutive page indices across the DRAM banks:
page `p` lives in bank `p % num_banks`. So the order in which one reader core
walks the page indices decides which banks its in-flight reads target — a purely
temporal, kernel-author decision, independent of where the core sits on the grid.

This example holds everything constant (same pages, same page size, same single
core, same block-of-reads-under-one-barrier policy) and varies ONLY the page-index
walk order (the stride between consecutive reads). Every variant reads the exact
same set of N pages, so the integer checksum is identical — only the order differs.

  - bank_stride  (baseline / the trap): stride == num_banks. Every read in a
    block lands on the SAME bank -> the bank serializes them, no cross-bank
    parallelism, poor row-buffer locality.
  - unit_stride  (candidate): stride == 1. Consecutive reads spread across all
    banks -> up to num_banks-way bank parallelism + good row-buffer locality.
  - coprime_stride (optional): stride coprime to num_banks (num_banks + 1). Also
    spreads across all banks (bank index steps by 1 mod num_banks) -> ~unit.

The walk is a general "coset" enumeration `idx = (base + k*stride) mod N` over
`g = gcd(stride, N)` cosets of length `N/g`; it visits every page exactly once for
any stride, so the read multiset (and checksum) is stride-independent. Reads are
issued a BLOCK at a time under one barrier so multiple are outstanding and the
bank parallelism can actually manifest. See README.md for the mechanism + numbers.
"""

from math import gcd
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

ELEM_BYTES = 2  # bfloat16 source element
ALIGN_BYTES = 32  # DRAM page alignment; page width must be a multiple of ALIGN_BYTES/ELEM_BYTES
ELEMS_PER_ALIGN = ALIGN_BYTES // ELEM_BYTES  # 16

CB_PAGES = 0  # reader scratch: `block` source pages read under one barrier
CB_OUT = 1  # reader scratch: one 32-byte checksum output page

# Baseline first: bank_stride is the trap, unit_stride is the fix, coprime_stride
# confirms it is bank-SPREAD (not contiguity) that matters.
VARIANTS = ("bank_stride", "unit_stride", "coprime_stride")


def num_dram_banks(device) -> int:
    """Number of interleaved DRAM banks (queried, never hard-coded)."""
    g = device.dram_grid_size()
    return g.x * g.y


def stride_for(variant: str, banks: int) -> int:
    """Map a named walk order to its concrete page-index stride, from the bank count."""
    if variant == "bank_stride":
        return banks  # every read hits the same bank
    if variant == "unit_stride":
        return 1  # consecutive reads spread across banks
    if variant == "coprime_stride":
        return banks + 1  # coprime to banks -> also spreads (steps banks by 1)
    raise ValueError(f"page_walk_order: unknown variant {variant!r}, expected one of {VARIANTS}")


def validate(source_tensor):
    """2D row-major bf16 source; each row is one interleaved DRAM page (width % 16 == 0)."""
    shape = list(source_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"page_walk_order: source must be rank 2 [N, W], got {shape}")
    if source_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("page_walk_order: source must be ROW_MAJOR_LAYOUT")
    if source_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"page_walk_order: source dtype must be bfloat16, got {source_tensor.dtype}")
    w = shape[1]
    if w % ELEMS_PER_ALIGN:
        raise ValueError(f"page_walk_order: page width ({w}) must be a multiple of {ELEMS_PER_ALIGN}")


def create_program_descriptor(source_tensor, output_tensor, *, stride, block, kernel_iters):
    device = source_tensor.device()
    num_pages = source_tensor.buffer_num_pages()  # one page == one row (row-major interleaved)
    if stride < 1:
        raise ValueError(f"page_walk_order: stride must be >= 1, got {stride}")

    g = gcd(stride, num_pages)
    coset_len = num_pages // g

    page_bytes = source_tensor.buffer_aligned_page_size()
    out_page_bytes = output_tensor.buffer_aligned_page_size()

    core = ttnn.CoreCoord(0, 0)  # single reader core: the cleanest picture of the walk order
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # cb_pages is fixed scratch of `block` pages: the reader reserves it, fills it with
    # a block of pipelined reads under ONE barrier, checksums it, and reuses it (never
    # pushed) for the next block. cb_out holds the single 32-byte checksum output page.
    cb_pages = ttnn.CBDescriptor(
        total_size=block * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_PAGES, data_format=ttnn.bfloat16, page_size=page_bytes)
        ],
    )
    cb_out = ttnn.CBDescriptor(
        total_size=out_page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=ttnn.uint32, page_size=out_page_bytes)
        ],
    )

    reader_ct_args = [page_bytes, num_pages, stride, g, coset_len, block, kernel_iters]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(source_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [source_tensor.buffer_address(), output_tensor.buffer_address()]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "pwo_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel], semaphores=[], cbs=[cb_pages, cb_out])


def page_walk_order(
    source_tensor: ttnn.Tensor,
    *,
    variant: str = "unit_stride",
    stride: int = None,
    block: int = None,
    kernel_iters: int = 1,
) -> ttnn.Tensor:
    """Walk every source page once in a chosen page-index order and return an integer checksum.

    Args:
        variant: the named walk order (see VARIANTS). Ignored if `stride` is given.
        stride: explicit page-index stride between consecutive reads (overrides `variant`).
        block: reads issued per barrier (how many are kept in flight). Default 2*num_banks
            — enough to expose full bank parallelism for the spreading walks.
        kernel_iters: in-kernel repeat of the whole N-page walk. 1 = per-launch latency,
            large = steady-state throughput. The checksum is reset each iter, so it is
            walk-order- and iters-independent.

    Output: a [1, 8] uint32 tensor whose word 0 is the order-independent checksum
    (integer sum of every source halfword) — identical for every walk order.
    """
    if kernel_iters < 1:
        raise ValueError(f"page_walk_order: kernel_iters must be >= 1, got {kernel_iters}")
    validate(source_tensor)
    device = source_tensor.device()
    banks = num_dram_banks(device)
    if stride is None:
        stride = stride_for(variant, banks)
    if block is None:
        block = 2 * banks
    if block < 1:
        raise ValueError(f"page_walk_order: block must be >= 1, got {block}")

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 8]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    program_descriptor = create_program_descriptor(
        source_tensor, output_tensor, stride=stride, block=block, kernel_iters=kernel_iters
    )
    return ttnn.generic_op([source_tensor, output_tensor], program_descriptor)
