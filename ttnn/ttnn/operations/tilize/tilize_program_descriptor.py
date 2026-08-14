# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor (CBs, kernels, args).

Blocking model (op_design.md §1): the work unit is a **block** =
1 tile-row x ``WT_CHUNK`` tile-columns. Every knob — ``CB_DEPTH``, ``NT_BLK``,
``WT_CHUNK``, ``NUM_CORES`` — is a parameter with a single source in
``derive_blocking()``; CB page counts, loop trip counts and grid sizing are all
computed *from* those knobs, never from a whole-op dimension.

Blocks are indexed W-chunk-major (``wc = b // NT_H``, ``row = b % NT_H``) so a
core's consecutive blocks share one W chunk and march linearly through the
source page ids.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

# --- fixed hardware / library facts (not knobs) ----------------------------
DEFAULT_TILE_WIDTH = 32  # a tile is always 32 wide
NUM_CIRCULAR_BUFFERS = 32

# --- blocking-model constants (single source; every knob derives from these)-
CB_L1_BUDGET = 1_048_576  # bytes of L1 reserved for the two streaming CBs
FAST_TILIZE_MAX_W = 255  # tilize_helpers.inl:95 -> block_width_tiles < 256

# --- block factors ----------------------------------------------------------
# NT_BLK: tile-rows per reader barrier-block. The library reader
# (read_sticks_for_tilize) barriers once per tile-row, so the Phase-0 value is 1
# and raising it needs a custom reader (op_design.md lamp L3). It is a named
# knob here — never an inlined literal — so the CB formula below is already
# written against it.
NT_BLK = 1

# --- CB indices (semantic names; the numeric slot is just a buffer index) ---
CB_INPUT_STICKS = 0  # reader -> compute  (row-major sticks, tile-sized pages)
CB_OUTPUT_TILES = 16  # compute -> writer (tiled pages)

# --- reader regime selector (op_design.md §5.1) -----------------------------
R_ALIGNED = 0
R_PAD = 1


def _prod(values):
    out = 1
    for v in values:
        out *= int(v)
    return out


def _div_up(a, b):
    return -(-a // b)


def derive_blocking(nt_h, wt, in_tile_bytes, out_tile_bytes, num_cores, cb_depth):
    """The three block knobs — single source of truth (op_design.md §1.4).

    Returns ``(wt_chunk, n_chunks, num_blocks)``.

    * ``WT_CHUNK`` is the COARSEST chunk that fits: the whole tile-row width
      unless the L1 ceiling or the grid-fill floor forces it smaller.
    * ``n_chunks`` divides ``WT`` exactly, so every block has the same width and
      there is exactly one compute kernel (no cliff-width variant).
    * ``NT_H >= NUM_CORES`` implies ``n_chunks == 1``, i.e. the wide-shape
      machinery is inert on tall shapes (byte-identical to a pure height split).
    """
    per_chunk_tile = cb_depth * (in_tile_bytes + out_tile_bytes)
    wt_cap = max(1, min(FAST_TILIZE_MAX_W, CB_L1_BUDGET // per_chunk_tile))

    n_want = max(1, _div_up(num_cores, nt_h))  # grid-fill floor
    n_want = max(n_want, _div_up(wt, wt_cap))  # L1 ceiling
    n_want = min(n_want, wt)  # can never split W finer than one tile-column

    n_chunks = next(c for c in range(n_want, wt + 1) if wt % c == 0)
    wt_chunk = wt // n_chunks
    return wt_chunk, n_chunks, nt_h * n_chunks


def _pack_pad_word(value, dtype):
    """The fill, packed in the **input** element format, in the low bytes of a word.

    The kernel replicates it across the store width, so a sub-word element fills
    correctly (a value written once per 32-bit word is invisible at 0 and
    garbage at any other fill).
    """
    if value is None:
        return 0
    if dtype == ttnn.float32:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if dtype == ttnn.bfloat16:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        # round-to-nearest-even on the truncated mantissa
        bits += 0x7FFF + ((bits >> 16) & 1)
        return (bits >> 16) & 0xFFFF
    if dtype in (ttnn.uint32, ttnn.int32):
        return int(value) & 0xFFFFFFFF
    if dtype == ttnn.uint16:
        return int(value) & 0xFFFF
    if dtype == ttnn.uint8:
        return int(value) & 0xFF
    raise ValueError(f"tilize: no pad-value packing for dtype {dtype}")


def create_program_descriptor(input_tensor, output_tensor, plan) -> ttnn.ProgramDescriptor:
    # ========== 1. TENSOR / TILE GEOMETRY =================================
    tile_h, tile_w = plan.tile_h, plan.tile_w
    elem_in = input_tensor.element_size()

    in_tile_bytes = tile_h * tile_w * elem_in  # row-major CB page (tile-sized)
    out_tile_bytes = output_tensor.buffer_page_size()  # tiled page (bf8b carries exponents)

    target = list(plan.target)
    nt_h = _prod(target[:-2]) * _div_up(target[-2], tile_h)  # total tile-rows
    wt = _div_up(target[-1], tile_w)  # total tile-columns

    # ========== 2. KNOBS + WORK DISTRIBUTION ==============================
    cb_depth = 2 if plan.use_double_buffer else 1

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    if plan.use_multicore:
        full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    else:
        full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    num_cores_available = full_grid.num_cores()

    wt_chunk, n_chunks, num_blocks_total = derive_blocking(
        nt_h, wt, in_tile_bytes, out_tile_bytes, num_cores_available, cb_depth
    )

    # never OOM: fall back to depth-1 rather than exceed the L1 budget
    while cb_depth > 1 and cb_depth * wt_chunk * (in_tile_bytes + out_tile_bytes) > CB_L1_BUDGET:
        cb_depth -= 1

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        blocks_per_core_1,
        blocks_per_core_2,
    ) = ttnn.split_work_to_cores(
        full_grid, num_blocks_total, True
    )  # row_wise=True (master.md A1)

    cores = ttnn.corerange_to_cores(all_cores, num_cores, True)

    # ========== 3. CIRCULAR BUFFERS =======================================
    # Both CBs are sized CB_DEPTH * NT_BLK * WT_CHUNK pages — a function of the
    # knobs only, never of WT / NT_H / any tensor dimension.
    cb_pages = cb_depth * NT_BLK * wt_chunk
    tile_descriptor = ttnn.TileDescriptor(tile_h, tile_w)

    cb_input_sticks_descriptor = ttnn.CBDescriptor(
        total_size=cb_pages * in_tile_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_STICKS,
                data_format=input_tensor.dtype,
                page_size=in_tile_bytes,
                tile=tile_descriptor,
            )
        ],
    )
    cb_output_tiles_descriptor = ttnn.CBDescriptor(
        total_size=cb_pages * out_tile_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=out_tile_bytes,
                tile=tile_descriptor,
            )
        ],
    )

    # ========== 4. KERNELS =================================================
    regime = R_PAD if plan.has_pad_region else R_ALIGNED

    in_shape = list(plan.in_shape)
    h_in = in_shape[-2]
    w_in_bytes = in_shape[-1] * elem_in
    n_img_in = _prod(in_shape[:-2])
    nth_per_img = _div_up(target[-2], tile_h)
    pad_word = _pack_pad_word(plan.pad_value, input_tensor.dtype)

    # -- reader (NCRISC / NOC0) --
    reader_ct_args = [
        regime,
        tile_h,
        wt_chunk,
        nt_h,
        nth_per_img,
        h_in,
        n_img_in,
        w_in_bytes,
        elem_in,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # -- writer (BRISC / NOC1) --
    writer_ct_args = [wt_chunk, nt_h, wt, out_tile_bytes]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    needs_cast = input_tensor.dtype != output_tensor.dtype
    compute_ct_args = [wt_chunk, 1 if needs_cast else 0]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    start_block = 0
    for core in cores:
        if core_group_1.contains(core):
            blocks_this_core = blocks_per_core_1
        elif core_group_2.contains(core):
            blocks_this_core = blocks_per_core_2
        else:
            blocks_this_core = 0
        reader_rt_args[core.x][core.y] = [src_addr, start_block, blocks_this_core, pad_word]
        writer_rt_args[core.x][core.y] = [dst_addr, start_block, blocks_this_core]
        compute_rt_args[core.x][core.y] = [blocks_this_core]
        start_block += blocks_this_core

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # fp32 -> fp32 must be BIT-EXACT: keep Dest in fp32 and stop the unpacker
    # downgrading fp32 to tf32 on its way to Dest. Only legal when the fast
    # tilize path is off (it is: fp32 OUTPUT disables it), which is exactly the
    # fp32-in/fp32-out case.
    lossless_fp32 = input_tensor.dtype == ttnn.float32 and output_tensor.dtype == ttnn.float32
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = lossless_fp32
    if lossless_fp32:
        unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
        unpack_modes[CB_INPUT_STICKS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_input_sticks_descriptor, cb_output_tiles_descriptor],
    )
