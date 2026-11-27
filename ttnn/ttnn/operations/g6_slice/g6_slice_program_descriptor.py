# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for g6_slice: one reader (NCRISC/NoC0) and one writer (BRISC/NoC1) per core.

Begins and buffer addresses are runtime args (common to every core); shapes, block size and
regime flags are compile-time args. The descriptor is built once per plan-cache key and a hit
rewrites only the two buffer-address slots (`patch_address`).
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_SLICE_PAGES = 0  # staging pages: reader -> writer
CB_RIGHT_COL_TILE = 1  # col_realign only: right input tile column, reader-private scratch

# Common runtime-arg slot of each kernel's buffer address (patched on a plan-cache hit).
READER_X_ADDR_SLOT = 0
WRITER_OUT_ADDR_SLOT = 0
READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1


def _cb(index, num_pages, page_bytes, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _patch_address(descriptor, input_tensor, output_tensor):
    descriptor.kernels[READER_KERNEL_INDEX].common_runtime_args[READER_X_ADDR_SLOT] = input_tensor.buffer_address()
    descriptor.kernels[WRITER_KERNEL_INDEX].common_runtime_args[WRITER_OUT_ADDR_SLOT] = output_tensor.buffer_address()


def create_program_descriptor(input_tensor, output_tensor, plan):
    """Returns (descriptor, patch_address)."""
    n4, b4, m4 = plan["n4"], plan["b4"], plan["m4"]
    page_bytes = plan["page_bytes"]
    block_pages = plan["block_pages"]
    cb_depth = plan["cb_depth"]
    all_cores = plan["all_cores"]
    regime = plan["regime"]

    row_runs = int(regime in ("row_runs", "col_realign"))
    w_shift = int(regime == "col_realign")

    tile = 32
    tensor_in_ht_tiles = -(-n4[2] // tile)
    tensor_in_wt_tiles = -(-n4[3] // tile)
    tensor_out_ht_tiles = -(-m4[2] // tile)
    tensor_out_wt_tiles = -(-m4[3] // tile)
    rows_last_tile_row = m4[2] - tile * (tensor_out_ht_tiles - 1)
    cols_last_tile_col = m4[3] - tile * (tensor_out_wt_tiles - 1)
    # The last tile column's right face half (cols 16..31) holds no logical column: leave it unmoved.
    # col_realign chooses its face halves from the column offset instead.
    left_half_last_col = int(not w_shift and cols_last_tile_col <= tile // 2)

    cbs = [_cb(CB_SLICE_PAGES, cb_depth * block_pages, page_bytes, input_tensor.dtype, all_cores)]
    if w_shift:
        cbs.append(_cb(CB_RIGHT_COL_TILE, block_pages, page_bytes, input_tensor.dtype, all_cores))

    reader_ct_args = [
        CB_SLICE_PAGES,
        CB_RIGHT_COL_TILE,
        page_bytes,
        block_pages,
        tensor_in_ht_tiles,
        tensor_in_wt_tiles,
        n4[1],
        tensor_out_ht_tiles,
        tensor_out_wt_tiles,
        m4[1],
        rows_last_tile_row,
        cols_last_tile_col,
        row_runs,
        w_shift,
        left_half_last_col,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        CB_SLICE_PAGES,
        page_bytes,
        block_pages,
        tensor_out_ht_tiles,
        tensor_out_wt_tiles,
        rows_last_tile_row,
        row_runs,
        left_half_last_col,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Begins (normalized, rank 4): lead dims as whole indices, the last two split into tile + in-tile offset.
    reader_common_rt = [
        input_tensor.buffer_address(),  # READER_X_ADDR_SLOT
        b4[0],
        b4[1],
        b4[2] // tile,  # first_input_tile_row
        b4[2] % tile,  # row_offset_in_tile
        b4[3] // tile,  # first_input_tile_col
        b4[3] % tile,  # col_offset_in_tile
    ]
    writer_common_rt = [
        output_tensor.buffer_address(),  # WRITER_OUT_ADDR_SLOT
        b4[2] % tile,  # row_offset_in_tile
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    pages_per_image = tensor_out_ht_tiles * tensor_out_wt_tiles
    for core, core_start_page, core_pages in plan["assignment"]:
        # core_start_coord (o0, o1, th, tw), decomposed here so the kernels never divide.
        tw = core_start_page % tensor_out_wt_tiles
        th = (core_start_page // tensor_out_wt_tiles) % tensor_out_ht_tiles
        image = core_start_page // pages_per_image
        o1 = image % m4[1]
        o0 = image // m4[1]
        reader_rt[core.x][core.y] = [core_pages, o0, o1, th, tw]
        writer_rt[core.x][core.y] = [core_start_page, core_pages, th, tw]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        common_runtime_args=reader_common_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        common_runtime_args=writer_common_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    kernels = [None, None]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return descriptor, _patch_address
