# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data-movement pipelining example: how to structure a DRAM eltwise op for throughput.

`double_buffer` runs a unary eltwise op (relu) over an interleaved DRAM tensor:
each core reads its tiles from DRAM (reader, NoC0), applies relu on the math
engine (compute), and writes them back to DRAM (writer, NoC1) — a classic
reader -> compute -> writer pipeline. It isolates TWO kernel-level data-movement
knobs that decide whether that pipeline is latency-bound or bandwidth-bound:

  1. READS/WRITES PER BARRIER (`block`). The reader issues `block` async reads
     back-to-back and then waits on ONE barrier (the writer is symmetric).
     block=1 is the trap: read-one / barrier / read-one / barrier is
     LATENCY-bound — every tile eats a full DRAM round trip before the next
     read even starts. block>1 keeps up to `block` transfers in flight so they
     pipeline, approaching DRAM BANDWIDTH. This is usually the bigger lever.

  2. DOUBLE BUFFERING (`variant`). The two CBs are `depth * block` tiles deep:
        variant="single_buffered" : depth=1 -> the reader must wait for the whole
                                    previous block to drain before refilling, and
                                    compute must wait for the writer to drain a
                                    block before producing more. Stages hand off
                                    in lockstep.
        variant="double_buffered" : depth=2 -> the reader can prefetch the next
                                    block while compute/writer drain the current
                                    one; the stages OVERLAP.

The reader / compute / writer kernels are BYTE-IDENTICAL for every `block` and
both variants — only compile-time args and the CB `total_size` change. So any
measured difference is attributable purely to those two knobs.

A third axis is TRANSFER SIZE: the input dtype sets the bytes moved per tile (the
NoC transaction size) — bfloat8_b ~1088 B, bfloat16 2048 B, float32 4096 B. The
kernels are dtype-agnostic; the CB page size is queried from the tensor
(`buffer_aligned_page_size`, never hard-coded) and only the compute
dest-accumulation mode changes (32-bit for float32). Pass an input of the dtype
you want to compare.

`compute_passes` repeats relu on each tile (relu is idempotent, so the result is
always relu(x)); it is a pure compute-cost knob to set how heavy compute is vs
the DRAM traffic. Keep it light (the default, 1) to study data movement.
See README.md for the mechanism, the ns + GB/s numbers, and the CLI.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_IN = 0  # reader -> compute
CB_OUT = 16  # compute -> writer

# Baseline first: single-buffered is the trap (no overlap), double-buffered is the fix.
VARIANTS = ("single_buffered", "double_buffered")

_DEPTH = {"single_buffered": 1, "double_buffered": 2}


# Supported tile formats — the "transfer size" axis. Each is a different number of
# bytes per tile (the NoC transaction size): bfloat8_b ~1088 B, bfloat16 2048 B,
# float32 4096 B. The kernels are dtype-agnostic; only the CB page size (queried
# from the tensor) and the compute dest-accumulation mode change.
SUPPORTED_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32)


def validate(input_tensor):
    """Keep the example dead simple: 2D, TILE, tile-aligned, interleaved; dtype in SUPPORTED_DTYPES."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"double_buffer example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("double_buffer example: input must be TILE_LAYOUT")
    if input_tensor.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"double_buffer example: dtype must be one of {SUPPORTED_DTYPES}, got {input_tensor.dtype}")
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"double_buffer example: H and W must be multiples of {TILE}, got {shape}")


def _resolve_num_cores(device, num_cores):
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    if num_cores is None:
        return 1  # single core is the cleanest picture: one pipeline, no cross-core NoC contention
    if num_cores < 1 or num_cores > max_cores:
        raise ValueError(f"double_buffer example: num_cores must be in [1, {max_cores}], got {num_cores}")
    return num_cores


def _ordered_cores(device, n):
    """`n` cores filled row-major across the grid. Both variants use the identical
    cores, so core placement is held constant and only the CB depth varies."""
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_pages(num_pages, n):
    """Contiguous page ranges by core index; remainder spread over the first cores.
    Identical for both variants -> identical per-core work."""
    base, rem = divmod(num_pages, n)
    ranges = []
    start = 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def create_program_descriptor(input_tensor, output_tensor, *, variant, num_cores, compute_passes, kernel_iters, block):
    if variant not in VARIANTS:
        raise ValueError(f"double_buffer example: variant must be one of {VARIANTS}, got {variant!r}")
    device = input_tensor.device()
    num_cores = _resolve_num_cores(device, num_cores)
    depth = _DEPTH[variant]

    page_bytes = (
        input_tensor.buffer_aligned_page_size()
    )  # per-tile transfer size (dtype-dependent: bfp8_b ~1088, bf16 2048, fp32 4096 B)
    num_pages = output_tensor.buffer_num_pages()
    assert num_pages == input_tensor.buffer_num_pages()
    if num_pages < num_cores:
        raise ValueError(f"double_buffer example: need >= {num_cores} pages, got {num_pages}")

    cores = _ordered_cores(device, num_cores)
    core_ranges = _core_range_set(cores)
    assignment = _assign_pages(num_pages, num_cores)

    # The two knobs under study, both expressed in the CB total_size:
    #   block  -> reader/writer reserve `block` tiles per barrier (reads in flight).
    #   depth  -> single_buffered (1) vs double_buffered (2): how many blocks the CB holds.
    # A CB must be at least `block` tiles deep for the reader to reserve a whole block.
    cb_tiles = depth * block
    cb_in = ttnn.CBDescriptor(
        total_size=cb_tiles * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )
    cb_out = ttnn.CBDescriptor(
        total_size=cb_tiles * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=output_tensor.dtype, page_size=page_bytes)
        ],
    )

    reader_ct_args = [page_bytes, kernel_iters, block]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [page_bytes, kernel_iters, block]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_ct_args = [kernel_iters, compute_passes]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, (start_page, count) in zip(cores, assignment):
        reader_rt_args[core.x][core.y] = [in_addr, start_page, count]
        writer_rt_args[core.x][core.y] = [out_addr, start_page, count]
        compute_rt_args[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "db_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "db_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # writer -> NoC1
    )
    # float32 tiles need a 32-bit dest register to pack without losing precision.
    fp32_dest_acc = input_tensor.dtype == ttnn.float32
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "db_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=[cb_in, cb_out]
    )


def double_buffer(
    input_tensor: ttnn.Tensor,
    *,
    variant: str = "double_buffered",
    block: int = 8,
    num_cores: int = None,
    compute_passes: int = 1,
    kernel_iters: int = 1,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Unary relu over an interleaved DRAM tensor as a reader->compute->writer pipeline.

    Args:
        variant: "single_buffered" (CB = 1 block deep, no prefetch, baseline) or
            "double_buffered" (CB = 2 blocks deep, stages overlap). Same kernels;
            only the CB depth differs.
        block: async reads/writes issued per NoC barrier. block=1 is the
            latency-bound trap (a full DRAM round trip per tile); larger keeps
            more transfers in flight and approaches DRAM bandwidth. Also sets the
            CB block granularity, so the CB holds `depth * block` tiles.
        num_cores: how many cores run the (independent) pipeline. Default 1 — the
            cleanest reading, with no cross-core NoC contention.
        compute_passes: how many times relu is applied per tile. relu is
            idempotent, so this is a pure compute-cost knob (output is always
            relu(x)); keep it light to study data movement.
        kernel_iters: in-kernel repeat of the whole tile range. 1 = per-launch
            latency, large = steady-state throughput.

    Output is relu(input) for every setting.
    """
    if block < 1:
        raise ValueError(f"double_buffer example: block must be >= 1, got {block}")
    if compute_passes < 1:
        raise ValueError(f"double_buffer example: compute_passes must be >= 1, got {compute_passes}")
    if kernel_iters < 1:
        raise ValueError(f"double_buffer example: kernel_iters must be >= 1, got {kernel_iters}")
    validate(input_tensor)
    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, device, out_mem
    )
    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        variant=variant,
        num_cores=num_cores,
        compute_passes=compute_passes,
        kernel_iters=kernel_iters,
        block=block,
    )
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
