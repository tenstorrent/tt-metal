# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor (CBs, kernels, args).

Data path (interleaved DRAM/L1 -> DRAM/L1):

    reader (NCRISC, NoC0): read 32 RM sticks = 1 tile-row -> cb_rm_in
      -> compute (TRISC): tilize_block reorders faces -> cb_tiled_out (pack cast)
      -> writer (BRISC, NoC1): drain Wt_chunk tile pages per tile-row -> output

Work is distributed along the tile-row (height) axis; each core owns a disjoint
range of tile-rows (no inter-core communication — tiles are independent).
Wide W is chunked so the CB footprint is bounded by a constant, not Wt.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_H = 32
TILE_W = 32

CB_RM_IN = 0  # reader -> compute (row-major, tile-sized pages)
CB_TILED_OUT = 16  # compute -> writer (tiled pages, output dtype)

# Cap on the per-block width so the CB footprint stays bounded on wide tensors.
WT_CHUNK_MAX = 8


def _pick_wt_chunk(wt: int) -> int:
    """Largest divisor of `wt` that is <= WT_CHUNK_MAX (keeps chunking even)."""
    if wt <= WT_CHUNK_MAX:
        return wt
    chunk = WT_CHUNK_MAX
    while wt % chunk != 0:
        chunk -= 1
    return chunk


def _row_wise_cores(grid: "ttnn.CoreCoord", num_cores: int):
    """`num_cores` cores filled row-by-row across the DRAM-facing (x) axis."""
    cores = []
    for y in range(grid.y):
        for x in range(grid.x):
            if len(cores) >= num_cores:
                return cores
            cores.append(ttnn.CoreCoord(x, y))
    return cores


def _assign_tile_rows(nt_h: int, num_cores: int):
    """Contiguous tile-row ranges by core index; remainder on the first cores."""
    base, rem = divmod(nt_h, num_cores)
    ranges = []
    start = 0
    for k in range(num_cores):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


# ---------------------------------------------------------------------------
# Sharded (zero-copy) path
# ---------------------------------------------------------------------------
def _shard_dims(mem_config):
    """(shard_h_folded, shard_w, grid) for a sharded mem config.

    Leading shard dims fold into height (rank-agnostic), so the local shard is a
    contiguous `shard_h_folded x shard_w` row-major block. nd specs live under
    `nd_shard_spec` (memory_layout == INTERLEAVED); legacy specs under
    `shard_spec` (memory_layout in {HEIGHT,WIDTH,BLOCK}).
    """
    if mem_config.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED:
        nd = mem_config.nd_shard_spec
        shape = list(nd.shard_shape)
        grid = nd.grid
    else:
        ss = mem_config.shard_spec
        shape = list(ss.shape)
        grid = ss.grid
    shard_w = shape[-1]
    shard_h = 1
    for d in shape[:-1]:
        shard_h *= d
    return shard_h, shard_w, grid


def _enumerate_cores(core_ranges: "ttnn.CoreRangeSet"):
    cores = []
    for cr in core_ranges.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                cores.append(ttnn.CoreCoord(x, y))
    return cores


def _create_sharded_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    """Same-spec sharded I/O: zero-copy, compute-only.

    Both CBs alias the local L1 shard buffers; the compute kernel tilizes each
    core's resident RM shard straight into its resident TILE shard. No reader,
    no writer, no DRAM/NoC traffic. Requires identical input/output shard spec
    (validated in tilize.validate()).
    """
    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    in_tile_size = ttnn.tile_size(in_dtype)
    out_tile_size = ttnn.tile_size(out_dtype)

    shard_h, shard_w, grid = _shard_dims(input_tensor.memory_config())
    wt = shard_w // TILE_W
    num_blocks = shard_h // TILE_H
    core_ranges = grid

    # Input CB aliased onto the RM input shard. cb_descriptor_from_sharded_tensor
    # inherits the tensor's page size (one-row-per-page for ROW_MAJOR); override
    # it to a whole tile so the tilize helper accounts in tiles while the
    # row-major bytes sit in the same L1 (established sharded-tilize aliasing —
    # see examples/compute_block_size._tile_paged_backed_cb).
    cb_rm_in_desc = ttnn.cb_descriptor_from_sharded_tensor(CB_RM_IN, input_tensor)
    in_fds = cb_rm_in_desc.format_descriptors
    in_fds[0].page_size = in_tile_size
    cb_rm_in_desc.format_descriptors = in_fds

    # Output CB aliased onto the TILE output shard (already tile-paged).
    cb_tiled_out_desc = ttnn.cb_descriptor_from_sharded_tensor(CB_TILED_OUT, output_tensor)
    out_fds = cb_tiled_out_desc.format_descriptors
    out_fds[0].page_size = out_tile_size
    cb_tiled_out_desc.format_descriptors = out_fds

    is_fp32_in = 1 if in_dtype == ttnn.float32 else 0
    compute_ct_args = [wt, num_blocks, is_fp32_in]

    fp32_dest = in_dtype == ttnn.float32 or out_dtype == ttnn.float32
    compute_config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest)
    if is_fp32_in:
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 32
        unpack_modes[CB_RM_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute_sharded.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=ttnn.RuntimeArgs(),
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[compute_kernel],
        semaphores=[],
        cbs=[cb_rm_in_desc, cb_tiled_out_desc],
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    use_multicore: bool,
) -> ttnn.ProgramDescriptor:
    # Same-spec sharded I/O -> zero-copy compute-only path.
    if input_tensor.memory_config().is_sharded() and output_tensor.memory_config().is_sharded():
        return _create_sharded_program_descriptor(input_tensor, output_tensor)

    device = input_tensor.device()

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    elem_size = input_tensor.element_size()

    # ROW_MAJOR input: page = one stick = full row of bytes; num_pages = total sticks.
    row_bytes_full = input_tensor.buffer_page_size()
    total_num_rows = input_tensor.buffer_num_pages()

    tile_row_bytes = TILE_W * elem_size
    wt = row_bytes_full // tile_row_bytes  # full width in tiles (W / 32)
    nt_h = total_num_rows // TILE_H  # number of tile-rows (folded height / 32)

    in_tile_size = ttnn.tile_size(in_dtype)
    out_tile_size = ttnn.tile_size(out_dtype)

    wt_chunk = _pick_wt_chunk(wt)
    num_chunks = wt // wt_chunk
    chunk_bytes = wt_chunk * tile_row_bytes

    # ---- work distribution across tile-rows ----
    if use_multicore:
        grid = device.compute_with_storage_grid_size()
        num_cores = min(nt_h, grid.x * grid.y)
    else:
        grid = ttnn.CoreCoord(1, 1)
        num_cores = 1

    cores = _row_wise_cores(grid, num_cores)
    assignment = _assign_tile_rows(nt_h, num_cores)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    # ---- circular buffers (double-buffered, bounded by Wt_chunk) ----
    cb_rm_in_desc = ttnn.CBDescriptor(
        total_size=2 * wt_chunk * in_tile_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RM_IN,
                data_format=in_dtype,
                page_size=in_tile_size,
            )
        ],
    )
    cb_tiled_out_desc = ttnn.CBDescriptor(
        total_size=2 * wt_chunk * out_tile_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_TILED_OUT,
                data_format=out_dtype,
                page_size=out_tile_size,
            )
        ],
    )

    # ---- kernels ----
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    # Reader (NCRISC / NoC0)
    reader_ct_args = [chunk_bytes, num_chunks]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_rt_args = ttnn.RuntimeArgs()

    # Writer (BRISC / NoC1)
    writer_ct_args = [out_tile_size, wt, wt_chunk, num_chunks]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()

    # Compute (TRISC)
    is_fp32_in = 1 if in_dtype == ttnn.float32 else 0
    compute_ct_args = [wt_chunk, num_chunks, is_fp32_in]
    compute_rt_args = ttnn.RuntimeArgs()

    for core, (start_tile_row, count) in zip(cores, assignment):
        reader_rt_args[core.x][core.y] = [in_addr, start_tile_row * TILE_H, count * TILE_H]
        writer_rt_args[core.x][core.y] = [out_addr, start_tile_row, count]
        compute_rt_args[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    fp32_dest = in_dtype == ttnn.float32 or out_dtype == ttnn.float32
    compute_config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest)
    if is_fp32_in:
        # Bit-exact fp32 tilize: the input CB must keep fp32 through the unpacker
        # into Dest (default mode downgrades fp32 -> tf32). Pairs with
        # Fp32Mode::Lossless in the compute kernel.
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 32
        unpack_modes[CB_RM_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_rm_in_desc, cb_tiled_out_desc],
    )
