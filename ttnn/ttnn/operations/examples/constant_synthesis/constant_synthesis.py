# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Constant-valued output: move the bytes from DRAM, or invent them on-core.

`constant_synthesis` materializes a large output region whose contents are a
single constant value. The obvious kernel reads a DRAM-resident constant tensor
and writes it out — paying to fetch bytes it could have generated locally. But a
constant needs no source: you can synthesize it on-core. This isolates ONE
kernel-level data-movement decision — where the source bytes come from:

  1. stream_from_dram (baseline): the reader streams every output page from a
     DRAM-resident constant tensor (one remote NoC read per page); the writer
     writes each page to DRAM. Real DRAM read traffic — the full read half of the
     roofline, paid for bytes that are all identical.

  2. synthesize (candidate): no reader DRAM traffic at all. The reader builds ONE
     output page of the constant in L1, once (a handful of local word stores),
     and the writer replicates that resident template to every output page. Zero
     source bytes read.

Both variants run the identical writer — one NoC write of a whole page per output
page, one barrier each — so the write half is common and never biases the
comparison. Same output shape, dtype, page size, constant value, and core
placement across variants; the ONLY difference is whether the output bytes are
READ from DRAM or INVENTED on-core.

The mechanism is DRAM bandwidth. When the move saturates DRAM (enough cores /
large enough output), the baseline pushes read+write bytes through the DRAM
controller while the candidate pushes only write bytes — so removing the read
side roughly halves the DRAM traffic and the latency with it.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

ELEM_BYTES = 2  # bfloat16

CB_DATA = 0  # one output page of the constant: reader -> writer

# Baseline first: stream_from_dram is the naive way, synthesize is the fix.
VARIANTS = ("stream_from_dram", "synthesize")

_SYNTH = {"stream_from_dram": 0, "synthesize": 1}


def _bf16_bits(value):
    """The 16-bit bit pattern of `value` in bfloat16 — the constant the kernel replicates."""
    import torch  # local import: torch must not be a module-level dependency of a ttnn package file

    return int(torch.tensor([value], dtype=torch.bfloat16).view(torch.uint16)[0].item())


def validate(output_shape):
    """Keep the example simple: rank-2 row-major output [R, W] (one page per row)."""
    if len(output_shape) != 2:
        raise ValueError(f"constant_synthesis: output must be rank 2, got {list(output_shape)}")


def _resolve_num_cores(device, num_cores):
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    if num_cores is None:
        return max_cores  # default: full grid — the DRAM-bandwidth-bound regime where the read side shows
    if num_cores < 1 or num_cores > max_cores:
        raise ValueError(f"constant_synthesis: num_cores must be in [1, {max_cores}], got {num_cores}")
    return num_cores


def _ordered_cores(device, n):
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_rows(num_rows, n):
    """Contiguous row ranges by core index; remainder spread over the first cores.
    Identical for both variants -> identical per-core write work."""
    base, rem = divmod(num_rows, n)
    ranges, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def create_program_descriptor(source_tensor, output_tensor, *, variant, value, num_cores, kernel_iters, block):
    if variant not in VARIANTS:
        raise ValueError(f"constant_synthesis: variant must be one of {VARIANTS}, got {variant!r}")
    if block < 1:
        raise ValueError(f"constant_synthesis: block must be >= 1, got {block}")
    device = output_tensor.device()
    num_cores = _resolve_num_cores(device, num_cores)
    synth = _SYNTH[variant]

    num_rows = output_tensor.buffer_num_pages()  # one page == one output row (row-major interleaved)
    assert num_rows == source_tensor.buffer_num_pages()
    if num_rows < num_cores:
        num_cores = num_rows  # never assign more cores than pages

    page_bytes = output_tensor.buffer_aligned_page_size()  # whole output row, read from the tensor
    value_lo = _bf16_bits(value)

    cores = _ordered_cores(device, num_cores)
    core_ranges = _core_range_set(cores)
    assignment = _assign_rows(num_rows, num_cores)

    # cb_data is 2 blocks deep (double-buffered) so the baseline reader can fetch
    # the next block of pages while the writer drains the current one, and so both
    # variants keep `block` writes in flight per barrier (NoC-bandwidth-bound, not
    # per-page-latency-bound). synthesize only needs one resident template page,
    # but the depth/config is identical across variants -> fair.
    cb_data = ttnn.CBDescriptor(
        total_size=2 * block * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_DATA, data_format=ttnn.bfloat16, page_size=page_bytes)
        ],
    )

    reader_ct_args = [page_bytes, value_lo, synth, kernel_iters, block]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(source_tensor).get_compile_time_args())
    writer_ct_args = [page_bytes, synth, kernel_iters, block]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    src_addr = source_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, (start_row, count) in zip(cores, assignment):
        reader_rt_args[core.x][core.y] = [src_addr, start_row, count]
        writer_rt_args[core.x][core.y] = [out_addr, start_row, count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "cs_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "cs_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # writer -> NoC1
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb_data])


def constant_synthesis(
    source_tensor: ttnn.Tensor,
    *,
    variant: str = "synthesize",
    value: float = 1.0,
    num_cores: int = None,
    kernel_iters: int = 1,
    block: int = 8,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Produce a constant-valued output [R, W] (== `value` everywhere) in bfloat16.

    Args:
        variant: "stream_from_dram" (baseline: read the constant from a DRAM
            tensor, one page per output page) or "synthesize" (candidate: invent
            one page in L1 and replicate it — zero DRAM reads). Same writer; only
            the reader's source of bytes differs.
        value: the constant that fills the whole output.
        num_cores: cores running the (independent) pipeline. Default = full grid,
            the DRAM-bandwidth-bound regime where removing the read side shows.
        kernel_iters: in-kernel repeat of the page range. 1 = per-launch latency,
            large = steady-state throughput.
        block: async reads/writes issued per NoC barrier (up to `block` transfers
            in flight). Keeps both variants NoC-bandwidth-bound, not per-page
            latency-bound. cb_data is sized 2*block pages (double-buffered).

    Every setting produces the same output: `value` in every element.
    """
    if kernel_iters < 1:
        raise ValueError(f"constant_synthesis: kernel_iters must be >= 1, got {kernel_iters}")
    validate(list(source_tensor.shape))
    device = source_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(source_tensor.shape)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, out_mem
    )
    program_descriptor = create_program_descriptor(
        source_tensor,
        output_tensor,
        variant=variant,
        value=value,
        num_cores=num_cores,
        kernel_iters=kernel_iters,
        block=block,
    )
    return ttnn.generic_op([source_tensor, output_tensor], program_descriptor)
