# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The hidden NoC alignment-residue tax on sub-page span reads.

`transfer_alignment` extracts one `span_bytes` sub-page span from every row of a
row-major DRAM tensor. The span starts at an arbitrary byte offset inside the row.
A `noc_async_read` can only move a range when the SOURCE byte address and the
DESTINATION byte address are congruent modulo the alignment window (the DRAM/L1
alignment granule, queried at runtime — nothing is hard-coded). When the span
start is NOT congruent with the destination residue you cannot express the read as
one transfer: you must round the source down to an alignment boundary, over-read
`span_bytes + residue` bytes into an aligned scratch buffer, then do a local L1
realign pass to move the useful `span_bytes` into place. The congruent case pays
neither the over-read nor the realign.

The non-obvious lesson: it is not merely "aligned is faster". A misaligned sub-page
read is *impossible to issue as a single transfer*, so the kernel silently pays an
over-read plus a per-span L1 realign — and you can dodge it entirely by arranging
the span offset (or the CB write-pointer) so source and destination share the same
alignment residue. This example isolates exactly that one decision.

  1. misaligned (baseline, the trap): the span start has a non-zero residue. The
     reader rounds the source address down to the alignment boundary, over-reads
     `span_bytes + residue` bytes into an aligned scratch CB, then realigns the
     `span_bytes` span into the destination. Over-read + one L1 pass, per span.

  2. aligned (candidate, the dodge): the span start is arranged alignment-congruent
     with the destination residue, so a single direct `noc_async_read` moves exactly
     `span_bytes` — no scratch, no realign.

The writer is byte-identical across both variants (it drains the extracted span and
writes it back), compute is identity, and both run on the same core placement, so any
measured delta is attributable purely to the read strategy. Every geometry constant
(residue, offsets, scratch size) is derived from the runtime-queried alignment, so the
example is arch-agnostic.
"""

from collections import namedtuple
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

ELEM_BYTES = 2  # bfloat16 — the span payload element

CB_SCRATCH = 0  # reader-internal aligned over-read scratch (misaligned path only)
CB_SPAN = 16  # extracted span, reader -> writer

# Baseline first: `misaligned` is the naive way, `aligned` is the dodge.
VARIANTS = ("misaligned", "aligned")

_ALIGNED_FLAG = {"misaligned": 0, "aligned": 1}

# Geometry derived entirely from the queried alignment window + the span size.
SpanGeometry = namedtuple(
    "SpanGeometry",
    [
        "align",  # alignment window in bytes (queried)
        "span_bytes",  # useful span payload per row
        "residue",  # non-zero residue used by the misaligned variant
        "s_aligned",  # source byte offset of the congruent span (residue 0)
        "s_misaligned",  # source byte offset of the non-congruent span (residue == `residue`)
        "over_read",  # bytes the misaligned path must over-read (span_bytes + residue)
        "row_bytes",  # padded source row page in bytes
        "row_elems",  # source row width in elements
        "width_elems",  # span width in elements
        "off_aligned_elems",  # element offset of the aligned span (for a torch reference)
        "off_misaligned_elems",  # element offset of the misaligned span (for a torch reference)
    ],
)


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def span_geometry(align, span_bytes):
    """Place both spans given the alignment window; all offsets stay arch-agnostic.

    The aligned span sits one full window into the row (residue 0). The misaligned
    span sits half a window further (residue = align // 2 — guaranteed non-zero and
    element-sized for any power-of-two window >= 2 * ELEM_BYTES). The row is padded so
    the misaligned over-read never runs past the page.
    """
    if span_bytes % ELEM_BYTES:
        raise ValueError(f"transfer_alignment: span_bytes ({span_bytes}) must be a multiple of {ELEM_BYTES}")
    residue = align // 2
    if residue % ELEM_BYTES:
        raise ValueError(f"transfer_alignment: alignment window {align} too small for element size {ELEM_BYTES}")
    s_aligned = align  # residue 0
    s_misaligned = align + residue  # residue == `residue`, non-zero
    over_read = span_bytes + residue
    row_bytes = _round_up(s_misaligned + span_bytes + align, align)
    return SpanGeometry(
        align=align,
        span_bytes=span_bytes,
        residue=residue,
        s_aligned=s_aligned,
        s_misaligned=s_misaligned,
        over_read=over_read,
        row_bytes=row_bytes,
        row_elems=row_bytes // ELEM_BYTES,
        width_elems=span_bytes // ELEM_BYTES,
        off_aligned_elems=s_aligned // ELEM_BYTES,
        off_misaligned_elems=s_misaligned // ELEM_BYTES,
    )


def validate(source_tensor, span_bytes, align):
    """2D row-major bf16 source whose row is wide enough for the padded geometry."""
    shape = list(source_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"transfer_alignment: source must be rank 2, got {shape}")
    if source_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("transfer_alignment: source must be ROW_MAJOR_LAYOUT")
    if source_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"transfer_alignment: source dtype must be bfloat16, got {source_tensor.dtype}")
    geom = span_geometry(align, span_bytes)
    if shape[1] < geom.row_elems:
        raise ValueError(
            f"transfer_alignment: source row ({shape[1]} elems) too narrow for span_bytes={span_bytes}; "
            f"need >= {geom.row_elems} elems"
        )


def _resolve_num_cores(device, num_cores):
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    if num_cores is None:
        return 1  # single core is the cleanest picture: one pipeline, no cross-core NoC contention
    if num_cores < 1 or num_cores > max_cores:
        raise ValueError(f"transfer_alignment: num_cores must be in [1, {max_cores}], got {num_cores}")
    return num_cores


def _ordered_cores(device, n):
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_rows(num_rows, n):
    """Contiguous row ranges by core index; remainder over the first cores. Identical for both variants."""
    base, rem = divmod(num_rows, n)
    ranges, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def create_program_descriptor(source_tensor, output_tensor, *, variant, num_cores, kernel_iters, span_bytes, align):
    if variant not in VARIANTS:
        raise ValueError(f"transfer_alignment: variant must be one of {VARIANTS}, got {variant!r}")
    device = source_tensor.device()
    num_cores = _resolve_num_cores(device, num_cores)
    geom = span_geometry(align, span_bytes)

    num_rows = source_tensor.buffer_num_pages()  # one page == one row (row-major interleaved)
    assert num_rows == output_tensor.buffer_num_pages()
    if num_rows < num_cores:
        raise ValueError(f"transfer_alignment: need >= {num_cores} rows, got {num_rows}")

    row_page_bytes = source_tensor.buffer_aligned_page_size()  # whole source row
    out_page_bytes = output_tensor.buffer_aligned_page_size()  # one extracted span

    cores = _ordered_cores(device, num_cores)
    core_ranges = _core_range_set(cores)
    assignment = _assign_rows(num_rows, num_cores)

    l1_align = ttnn.get_l1_alignment()
    # cb_span is 2 slots deep so the writer drains one span while the reader produces the
    # next (identical for both variants -> fair). Each slot carries the useful span plus one
    # alignment window of headroom for the congruence-landing offset. cb_scratch holds the
    # misaligned path's aligned over-read (span_bytes + up to two windows of slack); it is
    # allocated for BOTH variants so the L1 footprint is identical and never a variable.
    span_slot = _round_up(span_bytes + align, l1_align)
    scratch_slot = _round_up(span_bytes + 2 * align, l1_align)
    cb_span = ttnn.CBDescriptor(
        total_size=2 * span_slot,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_SPAN, data_format=ttnn.bfloat16, page_size=span_slot)
        ],
    )
    cb_scratch = ttnn.CBDescriptor(
        total_size=scratch_slot,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_SCRATCH, data_format=ttnn.bfloat16, page_size=scratch_slot)
        ],
    )

    reader_ct_args = [
        align,
        span_bytes,
        geom.s_aligned,
        geom.s_misaligned,
        geom.residue,
        row_page_bytes,
        kernel_iters,
        _ALIGNED_FLAG[variant],
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(source_tensor).get_compile_time_args())
    writer_ct_args = [span_bytes, align, out_page_bytes, kernel_iters]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    src_addr = source_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, (start_row, count) in zip(cores, assignment):
        reader_rt_args[core.x][core.y] = [src_addr, start_row, count]
        writer_rt_args[core.x][core.y] = [out_addr, start_row, count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ta_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ta_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # writer -> NoC1
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb_scratch, cb_span])


def transfer_alignment(
    source_tensor: ttnn.Tensor,
    *,
    variant: str = "aligned",
    span_bytes: int = 1024,
    num_cores: int = None,
    kernel_iters: int = 1,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Extract one `span_bytes` sub-page span from every row of a DRAM tensor.

    Args:
        variant: "misaligned" (baseline: non-congruent span -> over-read + L1 realign)
            or "aligned" (candidate: congruent span -> one direct read). Same writer;
            only the reader's read strategy and the span offset differ.
        span_bytes: useful bytes extracted per row (multiple of the element size).
        num_cores: cores running the (independent) pipeline. Default 1 — cleanest reading.
        kernel_iters: in-kernel repeat of the row range. 1 = per-launch latency,
            large = steady-state throughput.

    Output row r holds the variant's span of source row r (aligned and misaligned
    variants extract spans at different, per-variant offsets — each a correct sub-page
    span; see the tests for the torch references and the same-value byte-identity control).
    """
    if kernel_iters < 1:
        raise ValueError(f"transfer_alignment: kernel_iters must be >= 1, got {kernel_iters}")
    device = source_tensor.device()
    align = ttnn.get_dram_alignment()  # congruence window for DRAM-resident reads/writes
    validate(source_tensor, span_bytes, align)
    geom = span_geometry(align, span_bytes)

    num_rows = int(source_tensor.shape[0])
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([num_rows, geom.width_elems]), source_tensor.dtype, ttnn.ROW_MAJOR_LAYOUT, device, out_mem
    )
    program_descriptor = create_program_descriptor(
        source_tensor,
        output_tensor,
        variant=variant,
        num_cores=num_cores,
        kernel_iters=kernel_iters,
        span_bytes=span_bytes,
        align=align,
    )
    return ttnn.generic_op([source_tensor, output_tensor], program_descriptor)
