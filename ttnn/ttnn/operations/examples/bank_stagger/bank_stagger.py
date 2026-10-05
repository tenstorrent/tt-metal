# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""NoC-contention example: STAGGERING the per-core read/write order across DRAM banks.

Interleaved DRAM puts page p in bank `p % num_banks`. If every core reads its 32 ROW_MAJOR
rows in the same order and the cores' rows line up on the banks, each issue step puts the
whole grid on ONE bank. Two work splits cause it:
  1. width blocking : cores share the same 32 rows at different width offsets (any arch).
  2. height blocking: each core owns rows 32*k..32*k+31; with 8 banks 32 % 8 == 0, so every
                      core is on the same bank at each step (Blackhole only).

The fix rotates each core's issue order by a per-core offset. Each half is a compile-time
switch in its kernel; off compiles the plain in-order loop:
    stagger_reads  : reader issues row  (i + core % 32) % 32
    stagger_writes : writer issues tile (i + core % chunk_wt) % chunk_wt
Same transactions, sizes, count and L1 addresses either way. See README.md.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_IN = 0  # reader -> compute (ROW_MAJOR block)
CB_OUT = 16  # compute -> writer (tiles)
CB_DEPTH = 2  # blocks per CB; held constant across variants

# Named switch combinations for the sweep; baseline first.
VARIANTS = {
    "none": dict(stagger_reads=False, stagger_writes=False),
    "read": dict(stagger_reads=True, stagger_writes=False),
    "write": dict(stagger_reads=False, stagger_writes=True),
    "both": dict(stagger_reads=True, stagger_writes=True),
}

MAX_CHUNK_WT = 16


def validate(input_tensor, chunk_wt):
    """2D bf16 ROW_MAJOR interleaved; H a multiple of 32; W a multiple of 32*chunk_wt."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"bank_stagger example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("bank_stagger example: input must be ROW_MAJOR_LAYOUT")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"bank_stagger example: dtype must be bfloat16, got {input_tensor.dtype}")
    if input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("bank_stagger example: input must be interleaved")
    if not 1 <= chunk_wt <= MAX_CHUNK_WT:
        raise ValueError(f"bank_stagger example: chunk_wt must be in [1, {MAX_CHUNK_WT}], got {chunk_wt}")
    h, w = shape
    if h % TILE:
        raise ValueError(f"bank_stagger example: H must be a multiple of {TILE}, got {h}")
    if w % (TILE * chunk_wt):
        raise ValueError(f"bank_stagger example: W must be a multiple of {TILE * chunk_wt}, got {w}")


def work_geometry(shape, chunk_wt, grid_cores):
    """Work units are 32-row x chunk_wt-tile blocks, indexed row-major over
    (tile-row, chunk). Returns (nt_h, n_w, num_units, num_cores)."""
    h, w = shape
    nt_h, wt = h // TILE, w // TILE
    n_w = wt // chunk_wt
    units = nt_h * n_w
    return nt_h, n_w, units, min(units, grid_cores)


def _ordered_cores(device, n):
    """`n` cores filled row-major across the grid -- identical for every variant."""
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _split_units(units, n):
    """Contiguous unit ranges by core index; remainder on the first cores."""
    base, rem = divmod(units, n)
    ranges, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def num_dram_banks(device):
    return ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks


def create_program_descriptor(input_tensor, output_tensor, *, stagger_reads, stagger_writes, chunk_wt, kernel_iters=1):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    h, w = list(input_tensor.shape)
    _, n_w, units, num_cores = work_geometry((h, w), chunk_wt, grid.x * grid.y)

    page_bytes = input_tensor.buffer_aligned_page_size()  # one ROW_MAJOR row
    chunk_row_bytes = chunk_wt * TILE * 2  # bf16 slice of one row
    tile_bytes = output_tensor.buffer_aligned_page_size()

    cores = _ordered_cores(device, num_cores)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    assignment = _split_units(units, num_cores)

    def cb(index, dtype):
        return ttnn.CBDescriptor(
            total_size=CB_DEPTH * chunk_wt * tile_bytes,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
        )

    reader_ct = [chunk_wt, page_bytes, chunk_row_bytes, kernel_iters, int(stagger_reads)]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [chunk_wt, tile_bytes, kernel_iters, int(stagger_writes)]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_ct = [chunk_wt, kernel_iters]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for idx, (core, (start, count)) in enumerate(zip(cores, assignment)):
        row_rot, col_rot = idx % TILE, idx % chunk_wt  # read only when the switch is on
        reader_rt[core.x][core.y] = [in_addr, start, count, n_w, row_rot]
        writer_rt[core.x][core.y] = [out_addr, start, count, n_w, w // TILE, col_rot]
        compute_rt[core.x][core.y] = [count]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bs_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bs_writer.cpp"),
            core_ranges=core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bs_compute.cpp"),
            core_ranges=core_ranges,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(
        kernels=kernels, semaphores=[], cbs=[cb(CB_IN, input_tensor.dtype), cb(CB_OUT, output_tensor.dtype)]
    )


def bank_stagger(
    input_tensor: ttnn.Tensor,
    *,
    stagger_reads: bool = True,
    stagger_writes: bool = True,
    chunk_wt: int = 8,
    kernel_iters: int = 1,
) -> ttnn.Tensor:
    """Tilize a ROW_MAJOR interleaved DRAM bf16 tensor into an interleaved DRAM tile tensor.

    Args:
        stagger_reads: compile the reader with the rotated read order.
        stagger_writes: compile the writer with the rotated write order.
            Output is identical for every switch setting.
        chunk_wt: tile-columns per work unit (sets the read size: chunk_wt * 64 B per row).
        kernel_iters: in-kernel repeat of the unit range. 1 = per-launch latency.
    """
    if kernel_iters < 1:
        raise ValueError(f"bank_stagger example: kernel_iters must be >= 1, got {kernel_iters}")
    validate(input_tensor, chunk_wt)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        stagger_reads=stagger_reads,
        stagger_writes=stagger_writes,
        chunk_wt=chunk_wt,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
