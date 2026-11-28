# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast — ProgramDescriptor for regimes R1 `tile_stream` (TILE) and R2 `rm_pseudo_tile_stream`
(ROW_MAJOR) (op_design.md, Blocking Model). Selected by `input_tensor.layout`.

R2: the flat `row` axis (interleaved RM pages) is split into contiguous row ranges over the full grid.
Per block of `rows_per_block` rows the reader issues ONE NoC read per bank run (pages r, r+nb, r+2nb...
are adjacent in a bank at the aligned-page stride), compacts the rows densely into `cb_input_tiles`
(each 1024 values = one pseudo-tile), compute converts the pseudo-tiles with the same chain as R1
(unpack/pack apply the same intra-tile permutation, so an elementwise op preserves positions), and the
writer expands the dense rows into bank-run order and issues ONE NoC write per bank run.

Flat `tile` axis (batch x tile_row x tile_col folded, contiguous in page order) split into contiguous
ranges over the full compute grid (`split_work_to_cores(row_wise=True)`). Each core walks its range in
blocks of `block_tiles` tiles: one reader barrier, one CB handshake per kernel per block.

Knobs (single source each — every dependent quantity derives from them):
  BUFFER_DEPTH     — CB depth knob; effective depth = min(BUFFER_DEPTH, max blocks per core)
  BLOCK_TILES_CAP  — optional upper bound on block_tiles (None = coarsest block that fits L1)
  STAGING_DEPTH    — R2 only: staging slots per dataflow direction (2 = DMA / row-move overlap)
  RM_BLOCK_TILES   — R2 only: target pseudo-tiles per block (min'd with BLOCK_TILES_CAP)
  TILE_BLOCK_TILES — R1 only: target tiles per block (min'd with BLOCK_TILES_CAP)
  L1_OP_BUDGET_BYTES / L1_SLACK_BYTES — L1 grant (BB-05) and allocator headroom
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- knobs ----------------------------------------------------------------------------------------
BUFFER_DEPTH = 2
BLOCK_TILES_CAP = None  # None: block = whole per-core range clamped to the L1 grant
# R2 staging slots per direction: 2 overlaps the bank-run DMA of one block with the RISC-V compaction /
# expansion of the neighbouring block (reader and writer alike); 1 = serialized single staging zone.
STAGING_DEPTH = 2
# R2 block size in pseudo-tiles (None = coarsest block that fits L1). R2 is NoC/DRAM-bound (every RM row is
# a 64 B-aligned DRAM page); a small block shortens the unoverlapped pipeline fill (first read) and drain
# (last write). Sweep on F322/F317 (Blackhole, 110 cores): 1 -> 1568/384 us, 2 -> 1572/364 us,
# 3 -> 1604/379 us, 4 -> 1631/394 us, None (14) -> 1673/438 us. Bank runs stay >= rows/nb aligned pages.
RM_BLOCK_TILES = 2
# R1 block size in tiles (None = coarsest block that fits L1). The default whole-range block gave ONE block per
# core, so read -> convert -> write serialized on every core (effective depth 1). A small block pipelines the
# three stages. Sweep (Blackhole, 110 cores, depth 2; device-kernel us), cap None/16/8/6/4/3/2:
#   P39 [1,128,256,256] bf16->bf8b L1:  69.5/38.6/20.4/18.1/17.4/17.2/20.3(d4)
#   Q2185 [16,4096,12] bf16->fp32 DRAM: 36.6/37.5/35.2/35.7/34.6/35.0/34.9(d4)  (DRAM-write-bound)
# Depth 3 is within noise of depth 2 at block 3-4, so BUFFER_DEPTH stays 2 (keeps R2 unchanged).
TILE_BLOCK_TILES = 4
L1_OP_BUDGET_BYTES = 1347776
L1_SLACK_BYTES = 65536

# ---- CB indices (semantic names) ------------------------------------------------------------------
CB_INPUT_TILES = 0
CB_OUTPUT_TILES = 1
CB_INPUT_STAGING = 2  # R2 only: reader-private bank-run landing zone (aligned-page stride)
CB_OUTPUT_STAGING = 3  # R2 only: writer-private bank-run launch zone (aligned-page stride)

TILE_HW = 32
PSEUDO_TILE_VALUES = TILE_HW * TILE_HW
# NoC transfers need src/dst with the same offset modulo the DRAM read alignment (64 B on Blackhole);
# the staging zones are re-aligned to this inside the kernels (one extra unit allocated).
STAGING_ALIGN = 64

# 32-bit sources must unpack straight to DEST (SRCA keeps 19 bits); 32-bit or integer destinations need
# 32-bit DEST.
_SRC_32BIT = (ttnn.float32, ttnn.int32, ttnn.uint32)
# Integer sources (any width) also unpack straight to DEST: SRCA would reinterpret them as floats.
_SRC_UNPACK_TO_DEST = _SRC_32BIT + (ttnn.uint16,)
_DST_32BIT_DEST = (ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint8)
_NUM_CBS = 64

# Regime R0 `pure_copy` (Perf 1): arms whose source and destination bytes are identical, in both layouts.
BIT_IDENTICAL_ARMS = {(ttnn.uint32, ttnn.int32), (ttnn.int32, ttnn.uint32)}
# R0 L1 scratch per core (chunk mode: one buffer; page mode: two slots of half of it).
COPY_BUF_BYTES = 64 * 1024
# R0 chunk mode: bytes per core a page is split into when there are fewer pages than cores (knee measured on
# Q2220 / Q2189 / Q2226: more, smaller chunks lose to the per-core launch cost).
COPY_CHUNK_TARGET = 1024

# Runtime-arg slots patched on a program-cache hit (common runtime args, one per dataflow kernel).
_READER_KERNEL_IDX = 0
_WRITER_KERNEL_IDX = 1


def zones_enabled():
    """Stage zones (kernels/g6_typecast_zones.hpp) are compiled in only with G6_TYPECAST_ZONES=1."""
    return os.environ.get("G6_TYPECAST_ZONES", "0") not in ("", "0")


def _defines():
    return [("G6_TYPECAST_ZONES", "1")] if zones_enabled() else []


def _div_up(a, b):
    return (a + b - 1) // b


def _tensor_tiles(shape):
    """Tile count of a TILE tensor: prod(batch dims) * ceil(H/32) * ceil(W/32)."""
    w = shape[-1]
    h = shape[-2] if len(shape) >= 2 else 1
    batch = 1
    for d in shape[:-2]:
        batch *= d
    return batch * _div_up(h, TILE_HW) * _div_up(w, TILE_HW)


def _num_l1_banks(device):
    try:
        return device.num_banks(ttnn.BufferType.L1)
    except Exception:  # noqa: BLE001 — conservative fallback: one bank per worker core
        g = device.compute_with_storage_grid_size()
        return g.x * g.y


def plan_blocks(core_tiles_max, in_page_bytes, out_page_bytes, out_l1_bytes_per_bank):
    """Return (block_tiles, effective_depth): the coarsest block that fits the L1 grant."""
    cb_budget = L1_OP_BUDGET_BYTES - L1_SLACK_BYTES - out_l1_bytes_per_bank
    pair_bytes = in_page_bytes + out_page_bytes
    block_tiles = core_tiles_max
    for cap in (BLOCK_TILES_CAP, TILE_BLOCK_TILES):
        if cap is not None:
            block_tiles = min(block_tiles, cap)
    # Depth depends on the number of blocks, which depends on the block; iterate to a fixpoint.
    while True:
        depth = min(BUFFER_DEPTH, _div_up(core_tiles_max, block_tiles))
        cap = cb_budget // (depth * pair_bytes)
        assert cap >= 1, f"g6_typecast: L1 grant too small for one block ({cb_budget} B)"
        if block_tiles <= cap:
            return block_tiles, depth
        block_tiles = cap


def _compute_config(in_dtype, out_dtype):
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = out_dtype in _DST_32BIT_DEST or in_dtype in _SRC_32BIT
    compute_config.bfp8_pack_precise = out_dtype == ttnn.bfloat8_b
    compute_config.math_approx_mode = False
    if in_dtype in _SRC_UNPACK_TO_DEST:
        modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
        modes[CB_INPUT_TILES] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = modes
    return compute_config


def _tile_geometry(dtype, page_bytes):
    """(header bytes, face bytes) of one tile: bfloat8_b = 64 B of shared exponents then 4 x 256 B mantissa
    faces; every other format = 4 faces of page / 4 bytes, no header."""
    if dtype == ttnn.bfloat8_b:
        return 64, 256
    return 0, page_bytes // 4


def create_program_descriptor(input_tensor, output_tensor):
    if (input_tensor.dtype, output_tensor.dtype) in BIT_IDENTICAL_ARMS:
        return _create_copy_program_descriptor(input_tensor, output_tensor)
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        return _create_rm_program_descriptor(input_tensor, output_tensor)
    return _create_tile_program_descriptor(input_tensor, output_tensor)


def _create_tile_program_descriptor(input_tensor, output_tensor):
    device = input_tensor.device()
    shape = tuple(input_tensor.padded_shape)
    logical_shape = tuple(input_tensor.shape)
    out_dtype = output_tensor.dtype

    in_page_bytes = input_tensor.buffer_page_size()
    out_page_bytes = output_tensor.buffer_page_size()
    tensor_tiles = _tensor_tiles(logical_shape)
    assert tensor_tiles == input_tensor.buffer_num_pages() == output_tensor.buffer_num_pages(), (
        tensor_tiles,
        input_tensor.buffer_num_pages(),
        output_tensor.buffer_num_pages(),
    )

    # ---- work distribution: contiguous tile ranges over the full grid, row-wise ----
    grid = device.compute_with_storage_grid_size()
    (num_cores, all_cores, core_group_1, core_group_2, tiles_g1, tiles_g2) = ttnn.split_work_to_cores(
        grid, tensor_tiles, row_wise=True
    )
    core_tiles_max = tiles_g1

    # ---- block plan (L1 grant includes the op's own L1 output share) ----
    out_l1_bytes_per_bank = 0
    if output_tensor.memory_config().buffer_type == ttnn.BufferType.L1:
        out_l1_bytes_per_bank = _div_up(tensor_tiles, _num_l1_banks(device)) * out_page_bytes
    block_tiles, effective_depth = plan_blocks(core_tiles_max, in_page_bytes, out_page_bytes, out_l1_bytes_per_bank)
    cb_pages = effective_depth * block_tiles

    # ---- tile grid + tails: live-face transfers (Perf 1) and bf8b padding-column masking ----
    w = logical_shape[-1]
    h = logical_shape[-2] if len(logical_shape) >= 2 else 1
    w_tail = w % TILE_HW
    h_tail = h % TILE_HW
    tensor_tile_cols = _div_up(w, TILE_HW)
    tensor_tile_rows = _div_up(h, TILE_HW)
    mask_pad_columns = int(out_dtype == ttnn.bfloat8_b and w % 16 != 0)
    in_hdr, in_face = _tile_geometry(input_tensor.dtype, in_page_bytes)
    out_hdr, out_face = _tile_geometry(out_dtype, out_page_bytes)

    # ---- CBs ----
    cb_input_tiles = ttnn.CBDescriptor(
        total_size=cb_pages * in_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=in_page_bytes
            )
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=cb_pages * out_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUTPUT_TILES, data_format=out_dtype, page_size=out_page_bytes)
        ],
    )

    # ---- per-core RT args: [start_tile, core_tiles] ----
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    start_tile = 0
    for group, count in ((core_group_1, tiles_g1), (core_group_2, tiles_g2)):
        if count == 0 or group.num_cores() == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            reader_rt[core.x][core.y] = [start_tile, count]
            writer_rt[core.x][core.y] = [start_tile, count]
            compute_rt[core.x][core.y] = [count]
            start_tile += count
    assert start_tile == tensor_tiles

    # ---- kernels ----
    # RM-only CT slots (6..11) are inert zeros on the TILE path.
    reader_ct = [CB_INPUT_TILES, block_tiles, in_page_bytes, mask_pad_columns, tensor_tile_cols, w_tail]
    reader_ct += [0, CB_INPUT_STAGING, 0, 0, 0, STAGING_ALIGN, 1, tensor_tile_rows, h_tail, in_hdr, in_face]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[input_tensor.buffer_address()],
        defines=_defines(),
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct = [CB_OUTPUT_TILES, block_tiles, out_page_bytes, 0, CB_OUTPUT_STAGING, 0, 0, 0, STAGING_ALIGN, 1]
    writer_ct += [tensor_tile_cols, w_tail, tensor_tile_rows, h_tail, out_hdr, out_face]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address()],
        defines=_defines(),
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_config = _compute_config(input_tensor.dtype, out_dtype)
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=[CB_INPUT_TILES, CB_OUTPUT_TILES, block_tiles],
        runtime_args=compute_rt,
        defines=_defines(),
        config=compute_config,
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],  # order matches _READER_KERNEL_IDX / _WRITER_KERNEL_IDX
        semaphores=[],
        cbs=[cb_input_tiles, cb_output_tiles],
    )
    plan = {
        "regime": "tile_stream",
        "num_cores": num_cores,
        "tensor_tiles": tensor_tiles,
        "core_tiles_max": core_tiles_max,
        "block_tiles": block_tiles,
        "effective_depth": effective_depth,
        "mask_pad_columns": bool(mask_pad_columns),
    }
    return program_descriptor, plan


def plan_rm_blocks(
    core_rows_max, values_per_row, in_row_cost, out_row_cost, in_tile_bytes, out_tile_bytes, fixed_bytes
):
    """R2 block plan -> (rows_per_block, block_tiles, effective_depth).

    `block_tiles` (pseudo-tiles per block) stays the single block knob: `BLOCK_TILES_CAP` / `RM_BLOCK_TILES` cap it, and the
    rows of a block are the rows whose values fill at most `block_tiles` pseudo-tiles. Default: the coarsest
    block that fits the L1 grant. Per-row cost = STAGING_DEPTH x both staging slots (aligned pages); per-block cost =
    effective_depth x block_tiles x (in + out pseudo-tile bytes).
    """
    budget = L1_OP_BUDGET_BYTES - L1_SLACK_BYTES - fixed_bytes
    rows = core_rows_max
    caps = [c for c in (BLOCK_TILES_CAP, RM_BLOCK_TILES) if c is not None]
    if caps:
        rows = min(rows, max(1, (min(caps) * PSEUDO_TILE_VALUES) // values_per_row))
    while True:
        block_tiles = _div_up(rows * values_per_row, PSEUDO_TILE_VALUES)
        depth = min(BUFFER_DEPTH, _div_up(core_rows_max, rows))
        cost = rows * STAGING_DEPTH * (in_row_cost + out_row_cost) + depth * block_tiles * (
            in_tile_bytes + out_tile_bytes
        )
        if cost <= budget:
            return rows, block_tiles, depth
        assert rows > 1, f"g6_typecast: one row ({values_per_row} values) does not fit the L1 grant ({budget} B)"
        rows = max(1, min(rows - 1, (rows * budget) // cost))


def _create_rm_program_descriptor(input_tensor, output_tensor):
    """Regime R2 `rm_pseudo_tile_stream` (op_design.md, Regimes)."""
    device = input_tensor.device()
    logical_shape = tuple(input_tensor.shape)
    in_dtype, out_dtype = input_tensor.dtype, output_tensor.dtype

    values_per_row = logical_shape[-1]
    num_rows = input_tensor.buffer_num_pages()
    assert num_rows == output_tensor.buffer_num_pages(), (num_rows, output_tensor.buffer_num_pages())
    in_row_bytes = input_tensor.buffer_page_size()
    out_row_bytes = output_tensor.buffer_page_size()
    in_aligned = input_tensor.buffer_aligned_page_size()
    out_aligned = output_tensor.buffer_aligned_page_size()
    in_tile_bytes = (in_row_bytes // values_per_row) * PSEUDO_TILE_VALUES
    out_tile_bytes = (out_row_bytes // values_per_row) * PSEUDO_TILE_VALUES

    # ---- work distribution: contiguous row ranges over the full grid, row-wise ----
    grid = device.compute_with_storage_grid_size()
    (num_cores, all_cores, core_group_1, core_group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(
        grid, num_rows, row_wise=True
    )

    out_l1_bytes_per_bank = 0
    if output_tensor.memory_config().buffer_type == ttnn.BufferType.L1:
        out_l1_bytes_per_bank = _div_up(num_rows, _num_l1_banks(device)) * out_aligned
    staging_fixed = 2 * STAGING_ALIGN  # one re-alignment unit per staging CB
    rows_per_block, block_tiles, effective_depth = plan_rm_blocks(
        rows_g1,
        values_per_row,
        in_aligned,
        out_aligned,
        in_tile_bytes,
        out_tile_bytes,
        out_l1_bytes_per_bank + staging_fixed,
    )
    cb_pages = effective_depth * block_tiles
    # Staging slots never exceed the blocks a core walks (one block per core => one slot).
    staging_depth = min(STAGING_DEPTH, _div_up(rows_g1, rows_per_block))
    staging_rows = staging_depth * rows_per_block

    def _cb(index, dtype, page_bytes, pages):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
        )

    cbs = [
        _cb(CB_INPUT_TILES, in_dtype, in_tile_bytes, cb_pages),
        _cb(CB_OUTPUT_TILES, out_dtype, out_tile_bytes, cb_pages),
        # Staging zones: whole block of aligned pages + one alignment unit (re-aligned in-kernel).
        _cb(CB_INPUT_STAGING, in_dtype, in_aligned, staging_rows + _div_up(STAGING_ALIGN, in_aligned)),
        _cb(CB_OUTPUT_STAGING, out_dtype, out_aligned, staging_rows + _div_up(STAGING_ALIGN, out_aligned)),
    ]

    def _core_tiles(core_rows):
        full, rem = divmod(core_rows, rows_per_block)
        return full * block_tiles + _div_up(rem * values_per_row, PSEUDO_TILE_VALUES)

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    start_row = 0
    for group, count in ((core_group_1, rows_g1), (core_group_2, rows_g2)):
        if count == 0 or group.num_cores() == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            reader_rt[core.x][core.y] = [start_row, count]
            writer_rt[core.x][core.y] = [start_row, count]
            compute_rt[core.x][core.y] = [_core_tiles(count)]
            start_row += count
    assert start_row == num_rows

    # mask_pad_columns / tensor_tile_cols / w_tail are TILE-only (0 here).
    reader_ct = [CB_INPUT_TILES, block_tiles, in_aligned, 0, 1, 0]  # TILE-only mask / tile cols / w_tail: inert
    reader_ct += [1, CB_INPUT_STAGING, rows_per_block, in_row_bytes, values_per_row, STAGING_ALIGN, staging_depth]
    reader_ct += [1, 0, 1, 0]  # TILE-only tile rows / h_tail / tile header / face bytes: inert here
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[input_tensor.buffer_address()],
        defines=_defines(),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_ct = [CB_OUTPUT_TILES, block_tiles, out_aligned, 1, CB_OUTPUT_STAGING, rows_per_block, out_row_bytes]
    writer_ct += [values_per_row, STAGING_ALIGN, staging_depth]
    writer_ct += [1, 0, 1, 0, 0, 0]  # TILE-only tile grid / tails / tile geometry: inert here
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address()],
        defines=_defines(),
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=[CB_INPUT_TILES, CB_OUTPUT_TILES, block_tiles],
        runtime_args=compute_rt,
        defines=_defines(),
        config=_compute_config(in_dtype, out_dtype),
    )
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],  # order matches _READER_KERNEL_IDX / _WRITER_KERNEL_IDX
        semaphores=[],
        cbs=cbs,
    )
    plan = {
        "regime": "rm_pseudo_tile_stream",
        "num_cores": num_cores,
        "num_rows": num_rows,
        "core_rows_max": rows_g1,
        "rows_per_block": rows_per_block,
        "block_tiles": block_tiles,
        "effective_depth": effective_depth,
        "staging_depth": staging_depth,
    }
    return program_descriptor, plan


def _copy_units(num_pages, page_bytes, num_cores, grid_x, align):
    """R0 work split -> list of (page0, count, off, len), one per core.

    Page mode (pages >= cores): contiguous whole-page ranges. Chunk mode (fewer pages than cores): each page
    split into per_page = min(ceil(P / COPY_CHUNK_TARGET), cores // pages, grid.x) chunks of `align`-rounded
    bytes, one chunk per core."""
    if num_pages >= num_cores:
        base, rem = divmod(num_pages, num_cores)
        units, p = [], 0
        for m in range(num_cores):
            c = base + (m < rem)
            units.append((p, c, 0, page_bytes))
            p += c
        return units
    per_page = max(1, min(_div_up(page_bytes, COPY_CHUNK_TARGET), num_cores // num_pages, grid_x))
    chunk = _div_up(_div_up(page_bytes, per_page), align) * align
    return [(p, 1, off, min(chunk, page_bytes - off)) for p in range(num_pages) for off in range(0, page_bytes, chunk)]


def _bank_row_cores(device, units, dram):
    """Chunk-mode placement: the core of a chunk of page p sits on the NoC0-optimal worker row of p's DRAM bank
    (interleaved page p -> bank p % num_banks), nearest free row when that row is full. L1 buffers keep the
    row-major order from (0,0) (measured: (0,0) is already a fast core for L1)."""
    grid = device.compute_with_storage_grid_size()
    if not dram:
        return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(len(units))]
    num_banks = ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    best = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    used, cores = set(), []
    for page, *_ in units:
        y0 = best[page % num_banks].y
        for y in sorted(range(grid.y), key=lambda y: abs(y - y0)):
            free = [x for x in range(grid.x) if (x, y) not in used]
            if free:
                used.add((free[0], y))
                cores.append(ttnn.CoreCoord(free[0], y))
                break
    return cores


def _create_copy_program_descriptor(input_tensor, output_tensor):
    """Regime R0 `pure_copy` (Perf 1): bit-identical arms, any layout / rank / buffer. No compute kernel.

    Page mode (pages >= cores): reader (NCRISC/NoC0) + writer (BRISC/NoC1) per core on a 2-slot CB, whole pages.
    Chunk mode (pages < cores): one mover kernel (BRISC/NoC0) per core, one byte chunk of one page each,
    cores on the DRAM bank's NoC0-optimal row."""
    device = input_tensor.device()
    page_bytes = input_tensor.buffer_aligned_page_size()
    assert page_bytes == output_tensor.buffer_aligned_page_size()
    num_pages = input_tensor.buffer_num_pages()
    assert num_pages == output_tensor.buffer_num_pages()
    dram = input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    align = 64 if dram else 16
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    units = _copy_units(num_pages, page_bytes, num_cores, grid.x, align)
    chunk_mode = num_pages < num_cores
    if chunk_mode:
        cores = _bank_row_cores(device, units, dram)
    else:
        cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(len(units))]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    biggest = max(count * length for (_, count, _, length) in units)
    max_len = max(length for (*_, length) in units)
    slots = 1 if chunk_mode else 2
    slot = min(COPY_BUF_BYTES // slots, biggest)
    if max_len <= slot:
        slot = (slot // max_len) * max_len
    cb = ttnn.CBDescriptor(
        total_size=slots * slot,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=slot)
        ],
    )
    rt = ttnn.RuntimeArgs()
    for c, u in zip(cores, units):
        rt[c.x][c.y] = list(u)
    addrs = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    src_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    dst_ct = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    ct = [page_bytes, CB_INPUT_TILES, slot]
    if chunk_mode:
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "g6_typecast_copy_mover.cpp"),
                core_ranges=core_set,
                compile_time_args=ct + src_ct + dst_ct,
                runtime_args=rt,
                common_runtime_args=addrs,
                defines=_defines(),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            )
        ]
    else:
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "g6_typecast_copy_reader.cpp"),
                core_ranges=core_set,
                compile_time_args=ct + src_ct,
                runtime_args=rt,
                common_runtime_args=addrs,
                defines=_defines(),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "g6_typecast_copy_writer.cpp"),
                core_ranges=core_set,
                compile_time_args=ct + dst_ct,
                runtime_args=rt,
                common_runtime_args=addrs,
                defines=_defines(),
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
    program_descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb])
    plan = {
        "regime": "pure_copy",
        "num_cores": len(cores),
        "num_pages": num_pages,
        "chunk_mode": chunk_mode,
        "units": len(units),
        "slot_bytes": slot,
    }
    return program_descriptor, plan


def patch_buffer_addresses(program_descriptor, input_tensor, output_tensor, plan):
    """Program-cache hit: only the buffer addresses change (BB-07)."""
    if plan["regime"] == "pure_copy":
        addrs = [input_tensor.buffer_address(), output_tensor.buffer_address()]
        for kernel in program_descriptor.kernels:
            kernel.common_runtime_args = addrs
        return
    program_descriptor.kernels[_READER_KERNEL_IDX].common_runtime_args = [input_tensor.buffer_address()]
    program_descriptor.kernels[_WRITER_KERNEL_IDX].common_runtime_args = [output_tensor.buffer_address()]
