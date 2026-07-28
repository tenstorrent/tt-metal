# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Index-driven access: resolve random per-element indices in SRAM vs. one remote read each.

`index_staging` computes an indexed select `out[w] = src[idx[w]]` over the width
of a row-major DRAM tensor, where `idx` is an arbitrary index list. Each selected
element is a single bfloat16 value (2 bytes) — but the DRAM read granularity is a
whole aligned 32-byte line (ELEMS_PER_LINE elements): you cannot read 2 bytes
from an arbitrary DRAM offset. That mismatch is the whole point. It isolates ONE
kernel-level data-movement decision — how the random per-index access is serviced:

  1. remote_per_index (baseline): for each of the W indices, issue a SEPARATE
     remote NoC read of the whole aligned 32-byte line that CONTAINS element
     idx[w], then extract the 2 bytes wanted. W remote reads that each move 16x
     the needed bytes — the classic "waste a whole line for one element". Reads
     are pipelined (all issued, one barrier), so the penalty is transaction count
     and wasted bandwidth, not read-latency serialization.

  2. l1_staged (candidate): issue ONE bulk contiguous read of the whole source
     row into an L1 CB (each byte fetched once), then extract every element
     locally in SRAM. One remote transaction of exactly the useful bytes.

Both variants run the identical W-element local extract loop, so that cost is
common and never biases the comparison. The kernels are byte-identical except for
one compile-time flag; same shape, dtype, page size, and single-core placement
across variants — only the access strategy differs.

A second axis is INDEX DISTRIBUTION: `sorted` (monotonic) vs. `shuffled` (the same
index multiset, random order). It probes whether access order matters — see
README.md for the mechanism, the ns numbers, and the CLI.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# A selected element is one bfloat16 (2 bytes). The DRAM read granularity is a
# 32-byte aligned line (16 bf16), so a per-index remote read must pull a whole
# line to extract one element -> 16x wasted bytes. This mismatch is the concept.
ELEM_BYTES = 2  # bfloat16
ALIGN_BYTES = 32  # DRAM read/alignment granularity
ELEMS_PER_LINE = ALIGN_BYTES // ELEM_BYTES  # 16

CB_IDX = 0  # index row staged in L1 (reader-internal)
CB_LINE = 1  # baseline: W aligned scratch lines
CB_SRC = 2  # candidate: bulk source row
CB_OUT = 16  # selected output row: reader -> writer

# Baseline first: remote_per_index is the naive way, l1_staged is the fix.
VARIANTS = ("remote_per_index", "l1_staged")
DISTRIBUTIONS = ("sorted", "shuffled")

_STAGED = {"remote_per_index": 0, "l1_staged": 1}


def validate(source_tensor, index_tensor):
    """Keep the example simple: 2D row-major; source bf16 [R, W] (W % ELEMS_PER_LINE == 0); index uint32 [R, W]."""
    src_shape = list(source_tensor.shape)
    idx_shape = list(index_tensor.shape)
    if len(src_shape) != 2 or len(idx_shape) != 2:
        raise ValueError(f"index_staging: source and index must be rank 2, got {src_shape} and {idx_shape}")
    if source_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or index_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("index_staging: source and index must be ROW_MAJOR_LAYOUT")
    if source_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"index_staging: source dtype must be bfloat16, got {source_tensor.dtype}")
    if index_tensor.dtype != ttnn.uint32:
        raise ValueError(f"index_staging: index dtype must be uint32, got {index_tensor.dtype}")
    if src_shape != idx_shape:
        raise ValueError(f"index_staging: source and index shapes must match, got {src_shape} vs {idx_shape}")
    w = src_shape[1]
    if w % ELEMS_PER_LINE:
        raise ValueError(f"index_staging: W ({w}) must be a multiple of {ELEMS_PER_LINE} (line-aligned rows)")


def _resolve_num_cores(device, num_cores):
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    if num_cores is None:
        return 1  # single core is the cleanest picture: one pipeline, no cross-core NoC contention
    if num_cores < 1 or num_cores > max_cores:
        raise ValueError(f"index_staging: num_cores must be in [1, {max_cores}], got {num_cores}")
    return num_cores


def _ordered_cores(device, n):
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_rows(num_rows, n):
    """Contiguous row ranges by core index; remainder spread over the first cores.
    Identical for both variants -> identical per-core work."""
    base, rem = divmod(num_rows, n)
    ranges, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def create_program_descriptor(source_tensor, index_tensor, output_tensor, *, variant, num_cores, kernel_iters):
    if variant not in VARIANTS:
        raise ValueError(f"index_staging: variant must be one of {VARIANTS}, got {variant!r}")
    device = source_tensor.device()
    num_cores = _resolve_num_cores(device, num_cores)
    staged = _STAGED[variant]

    num_rows = source_tensor.buffer_num_pages()  # one page == one row (row-major interleaved)
    assert num_rows == index_tensor.buffer_num_pages() == output_tensor.buffer_num_pages()
    if num_rows < num_cores:
        raise ValueError(f"index_staging: need >= {num_cores} rows, got {num_rows}")

    w = index_tensor.shape[1]
    src_page_bytes = source_tensor.buffer_aligned_page_size()  # whole source row
    idx_page_bytes = index_tensor.buffer_aligned_page_size()  # whole index row
    out_page_bytes = output_tensor.buffer_aligned_page_size()

    cores = _ordered_cores(device, num_cores)
    core_ranges = _core_range_set(cores)
    assignment = _assign_rows(num_rows, num_cores)

    # cb_out is 2 rows deep so the writer can drain one selected row while the
    # reader produces the next (identical for both variants -> fair). cb_idx,
    # cb_line and cb_src are reader-internal staging. cb_line holds W aligned
    # lines (the baseline's wasteful scratch); cb_src holds one bulk source row.
    cb_idx = ttnn.CBDescriptor(
        total_size=idx_page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_IDX, data_format=ttnn.uint32, page_size=idx_page_bytes)
        ],
    )
    line_scratch_bytes = w * ALIGN_BYTES
    cb_line = ttnn.CBDescriptor(
        total_size=line_scratch_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_LINE, data_format=ttnn.bfloat16, page_size=line_scratch_bytes)
        ],
    )
    cb_src = ttnn.CBDescriptor(
        total_size=src_page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_SRC, data_format=ttnn.bfloat16, page_size=src_page_bytes)
        ],
    )
    cb_out = ttnn.CBDescriptor(
        total_size=2 * out_page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=ttnn.bfloat16, page_size=out_page_bytes)
        ],
    )

    reader_ct_args = [src_page_bytes, idx_page_bytes, ELEM_BYTES, ALIGN_BYTES, w, kernel_iters, staged]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(source_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(index_tensor).get_compile_time_args())
    writer_ct_args = [out_page_bytes, kernel_iters]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    src_addr = source_tensor.buffer_address()
    idx_addr = index_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, (start_row, count) in zip(cores, assignment):
        reader_rt_args[core.x][core.y] = [src_addr, idx_addr, start_row, count]
        writer_rt_args[core.x][core.y] = [out_addr, start_row, count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "is_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "is_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # writer -> NoC1
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb_idx, cb_line, cb_src, cb_out]
    )


def index_staging(
    source_tensor: ttnn.Tensor,
    index_tensor: ttnn.Tensor,
    *,
    variant: str = "l1_staged",
    num_cores: int = None,
    kernel_iters: int = 1,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Indexed select out[r, w] = source[r, index[r, w]] (per bfloat16 element) along each row.

    Args:
        variant: "remote_per_index" (baseline: W separate remote line reads per
            row) or "l1_staged" (candidate: one bulk row read + SRAM-local
            extract). Same kernels; only one compile-time flag differs.
        num_cores: how many cores run the (independent) pipeline. Default 1 — the
            cleanest reading, with no cross-core NoC contention.
        kernel_iters: in-kernel repeat of the whole row range. 1 = per-launch
            latency, large = steady-state throughput.

    Output row r holds source[r, index[r, :]] for every setting.
    """
    if kernel_iters < 1:
        raise ValueError(f"index_staging: kernel_iters must be >= 1, got {kernel_iters}")
    validate(source_tensor, index_tensor)
    device = source_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(source_tensor.shape)), source_tensor.dtype, ttnn.ROW_MAJOR_LAYOUT, device, out_mem
    )
    program_descriptor = create_program_descriptor(
        source_tensor,
        index_tensor,
        output_tensor,
        variant=variant,
        num_cores=num_cores,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([source_tensor, index_tensor, output_tensor], program_descriptor)
