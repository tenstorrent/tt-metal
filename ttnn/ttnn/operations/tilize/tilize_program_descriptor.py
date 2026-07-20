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


def _num_shards(tensor_shape, mem_config):
    """Number of shards (product of per-dim ceil divisions, rank-aligned from
    the right; legacy 2D specs cover the last-2-dims-folded view)."""
    if mem_config.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED:
        shard = list(mem_config.nd_shard_spec.shard_shape)
    else:
        shard = list(mem_config.shard_spec.shape)
    ts = list(tensor_shape)
    if len(shard) < len(ts):
        h = 1
        for d in ts[:-1]:
            h *= d
        ts = [h, ts[-1]]
    k = min(len(shard), len(ts))
    n = 1
    for t, s in zip(ts[-k:], shard[-k:]):
        n *= -(-t // s)  # ceil
    return n


def _pd_same_shard_spec(in_mc, out_mc):
    """True iff input and output describe the same PHYSICAL shard placement
    (buffer_type, folded shard shape, orientation, grid) — invariant to the
    nd<->legacy normalization ttnn applies. Mirrors tilize._same_shard_spec."""

    def props(mc):
        if mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED:
            spec = mc.nd_shard_spec
        else:
            spec = mc.shard_spec
        sh, sw, grid = _shard_dims(mc)
        return (mc.buffer_type, (sh, sw), spec.orientation, grid)

    return props(in_mc) == props(out_mc)


def _physical_num_blocks(tensor_shape, mem_config, shard_h):
    """Tile-rows in the PHYSICAL per-core bank (uniform across cores).

    A sharded buffer reserves `ceil(n_shards / n_cores)` shard slots on every
    core (uniform bank size; cliff cores' extra slots hold padding). Each shard
    contributes `shard_h / 32` tile-rows, so the whole bank is
    `ceil(n_shards/n_cores) * shard_h/32` tile-rows. For same-spec I/O the input
    and output banks share this layout slot-for-slot, so tilizing the full
    physical bank (padding included) preserves identity — the padded slots
    round-trip to the same padded slots and to_torch strips them on readback.
    """
    _, _, grid = _shard_dims(mem_config)
    n_cores = grid.num_cores()
    n_shards = _num_shards(tensor_shape, mem_config)
    max_shards_per_core = -(-n_shards // n_cores)  # ceil
    return max_shards_per_core * (shard_h // TILE_H)


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
    core_ranges = grid

    # num_blocks is derived from the PHYSICAL per-core bank (not a single shard),
    # so it spans every shard a core owns. The RM input buffer's page is one
    # shard-width row, so buffer_num_pages == total rows across all shards; a core
    # owns `buffer_num_pages / num_cores` contiguous rows. When a core holds k
    # shards (same-spec, even, no-padding — gated in validate()), those k shards
    # sit as one contiguous (k*shard_h) x shard_w RM block and tilize as
    # k*(shard_h/32) tile-rows of Wt tiles straight into the concatenated output
    # bank in shard-index order (identity preserved because both sides use the
    # IDENTICAL spec). For one-shard-per-core this reduces to shard_h/32.
    # num_blocks spans the PHYSICAL per-core bank (every shard slot the core
    # owns, padded cliff slots included). buffer_num_pages counts LOGICAL
    # (width-split) rows, which under-counts padded/cliff banks — so derive from
    # the physical shard geometry instead. Same-spec identity holds because the
    # input and output banks share this slot layout.
    num_blocks = _physical_num_blocks(list(input_tensor.shape), input_tensor.memory_config(), shard_h)

    # Input CB aliased onto the RM input shard. cb_descriptor_from_sharded_tensor
    # inherits the tensor's page size (one-row-per-page for ROW_MAJOR); override
    # it to a whole tile so the tilize helper accounts in tiles while the
    # row-major bytes sit in the same L1 (established sharded-tilize aliasing —
    # see examples/compute_block_size._tile_paged_backed_cb).
    # Pass core_ranges explicitly: genuinely-nd tensors (3+ dim shard, no legacy
    # 2D equivalent) have no `tensor.shard_spec()`, so the binding's default
    # `tensor.shard_spec()->grid` dereferences null. `grid` from _shard_dims is
    # the nd/legacy grid either way.
    cb_rm_in_desc = ttnn.cb_descriptor_from_sharded_tensor(CB_RM_IN, input_tensor, core_ranges=grid)
    in_fds = cb_rm_in_desc.format_descriptors
    in_fds[0].page_size = in_tile_size
    cb_rm_in_desc.format_descriptors = in_fds

    # Output CB aliased onto the TILE output shard (already tile-paged).
    cb_tiled_out_desc = ttnn.cb_descriptor_from_sharded_tensor(CB_TILED_OUT, output_tensor, core_ranges=grid)
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


# ---------------------------------------------------------------------------
# fp32 compute config (shared by the general cross-core path)
# ---------------------------------------------------------------------------
def _fp32_compute_config(in_dtype, out_dtype):
    is_fp32_in = 1 if in_dtype == ttnn.float32 else 0
    fp32_dest = in_dtype == ttnn.float32 or out_dtype == ttnn.float32
    cfg = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest)
    if is_fp32_in:
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 32
        unpack_modes[CB_RM_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = unpack_modes
    return is_fp32_in, cfg


# ---------------------------------------------------------------------------
# General cross-core sharded / crossover path (native default-factory model)
# ---------------------------------------------------------------------------
# Handles every sharded case that is NOT same-spec zero-copy: interleaved<->nd/
# WIDTH/BLOCK crossover, cross-spec resharding (nd->nd different spec), and
# nd->legacy. There is NO cross-core semaphore/multicast — work is split across
# the compute grid by OUTPUT tile-rows, and each compute core (a) reads the
# full-width RM sticks for its tile-rows from the input via TensorAccessor
# (resolving to DRAM or remote L1 shard banks), (b) tilizes, (c) writes its Wt
# output tiles per tile-row via TensorAccessor to the output (DRAM or remote L1
# shard banks). The accessors' logical page ordering (input: row*npr+chunk;
# output: tile-row-major page = tr*Wt + tc) is what makes any-to-any placement
# work with zero DRAM staging. Reuses tilize_compute.cpp + tilize_writer.cpp.
def _create_general_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    use_multicore: bool,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    elem_size = input_tensor.element_size()
    in_tile_size = ttnn.tile_size(in_dtype)
    out_tile_size = ttnn.tile_size(out_dtype)

    shape = list(input_tensor.shape)
    w = shape[-1]
    folded_h = 1
    for d in shape[:-1]:
        folded_h *= d
    wt = w // TILE_W
    nt_h = folded_h // TILE_H
    row_bytes = w * elem_size

    # Input read geometry: interleaved => one full-width page per row; sharded =>
    # ceil(W / shard_width) width-chunk pages per row (last chunk padded).
    in_mc = input_tensor.memory_config()
    if in_mc.is_sharded():
        _, in_shard_w, _ = _shard_dims(in_mc)
        npr = -(-w // in_shard_w)  # ceil
        chunk_bytes = in_shard_w * elem_size
    else:
        npr = 1
        chunk_bytes = row_bytes

    # Work split across the compute grid by output tile-rows (independent work —
    # each core reads its own input rows and writes its own output tiles).
    if use_multicore:
        grid = device.compute_with_storage_grid_size()
        num_cores = min(nt_h, grid.x * grid.y)
    else:
        grid = ttnn.CoreCoord(1, 1)
        num_cores = 1
    cores = _row_wise_cores(grid, num_cores)
    assignment = _assign_tile_rows(nt_h, num_cores)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    # Streaming CBs, double-buffered, bounded by Wt (small for the sharded golden
    # shapes; wide-W HEIGHT crossover chunking is a perf follow-up, Refinement 2d).
    cb_rm_in_desc = ttnn.CBDescriptor(
        total_size=2 * wt * in_tile_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_RM_IN, data_format=in_dtype, page_size=in_tile_size)
        ],
    )
    cb_tiled_out_desc = ttnn.CBDescriptor(
        total_size=2 * wt * out_tile_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_TILED_OUT, data_format=out_dtype, page_size=out_tile_size)
        ],
    )

    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    reader_ct_args = [wt, row_bytes, chunk_bytes, npr]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [out_tile_size, wt, wt, 1]  # Wt, Wt_chunk=Wt, num_chunks=1
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    is_fp32_in, compute_config = _fp32_compute_config(in_dtype, out_dtype)
    compute_ct_args = [wt, 1, is_fp32_in]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, (start_tile_row, count) in zip(cores, assignment):
        reader_rt_args[core.x][core.y] = [in_addr, start_tile_row * TILE_H, count * TILE_H]
        writer_rt_args[core.x][core.y] = [out_addr, start_tile_row, count]
        compute_rt_args[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader_general.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_rm_in_desc, cb_tiled_out_desc],
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    use_multicore: bool,
) -> ttnn.ProgramDescriptor:
    in_mc = input_tensor.memory_config()
    out_mc = output_tensor.memory_config()
    in_sharded = in_mc.is_sharded()
    out_sharded = out_mc.is_sharded()
    # Same-spec sharded I/O + multicore -> zero-copy compute-only path (fastest:
    # both CBs aliased onto resident L1 shards, no NoC).
    if in_sharded and out_sharded and use_multicore and _pd_same_shard_spec(in_mc, out_mc):
        return _create_sharded_program_descriptor(input_tensor, output_tensor)
    # Any other sharded case (crossover either direction, cross-spec both-sharded,
    # nd->legacy, single-core sharded) -> general cross-core path via TensorAccessor.
    if in_sharded or out_sharded:
        return _create_general_program_descriptor(input_tensor, output_tensor, use_multicore)

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
