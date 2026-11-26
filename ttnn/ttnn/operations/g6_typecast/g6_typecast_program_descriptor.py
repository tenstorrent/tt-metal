# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast program descriptor — regime `tile_convert`.

Page-to-page streaming over the flattened page axis `tile_page`: contiguous page ranges
spread over the full compute grid, `block_tiles` pages per block, double-buffered CBs.
Reader (NCRISC/NoC0) -> cb_input_tiles -> compute (typecast chain) -> cb_output_tiles ->
writer (BRISC/NoC1).
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Blocking-model knobs (op_design.md, Blocking Model). Every CB capacity and kernel loop bound
# derives from these.
BLOCK_TILES_CAP = 8  # pages per block; the overlap lamp's plateau (catalog double_buffer)
IN_DEPTH = 2  # cb_input_tiles depth in blocks
OUT_DEPTH = 2  # cb_output_tiles depth in blocks
PAGE_STRIDE = 1  # page walk stride; the deferred bank-local regime passes the L1 bank count

CB_INPUT_TILES = 0
CB_OUTPUT_TILES = 1

REGIME_TILE_CONVERT = "tile_convert"
BITCOPY_PAIRS = {(ttnn.uint32, ttnn.int32), (ttnn.int32, ttnn.uint32)}

_WIDE_DEST_DTYPES_IN = (ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16)
_WIDE_DEST_DTYPES_OUT = (ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint8)


def select_regime(layout, in_dtype, out_dtype):
    if layout == ttnn.TILE_LAYOUT and (in_dtype, out_dtype) not in BITCOPY_PAIRS:
        return REGIME_TILE_CONVERT
    raise NotImplementedError(f"g6_typecast: no built regime for {layout}, {in_dtype} -> {out_dtype}")


def plan_work(device, tensor_pages):
    """The split of `tile_page` over the full grid and the block knob derived from it."""
    grid = device.compute_with_storage_grid_size()
    num_cores, all_cores, core_group_1, core_group_2, core_tiles_g1, core_tiles_g2 = ttnn.split_work_to_cores(
        grid, tensor_pages, row_wise=True
    )
    # One ordered core list drives both first_page accumulation and RT-arg placement.
    assignment = []
    first_page = 0
    for group, core_tiles in ((core_group_1, core_tiles_g1), (core_group_2, core_tiles_g2)):
        if core_tiles == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            assignment.append((core, first_page, core_tiles))
            first_page += core_tiles
    assert first_page == tensor_pages, f"page ranges cover {first_page} of {tensor_pages} pages"
    block_tiles = min(core_tiles_g1, BLOCK_TILES_CAP)
    return {
        "grid": grid,
        "num_cores": num_cores,
        "all_cores": all_cores,
        "assignment": assignment,
        "block_tiles": block_tiles,
    }


def _compute_config(in_dtype, out_dtype):
    config = ttnn.ComputeConfigDescriptor()
    # Float32 / Int32 / UInt32 endpoints of typecast_tile need 32-bit DEST.
    config.fp32_dest_acc_en = in_dtype in _WIDE_DEST_DTYPES_IN or out_dtype in _WIDE_DEST_DTYPES_OUT
    config.math_approx_mode = False
    config.dst_full_sync_en = False
    if in_dtype in _WIDE_DEST_DTYPES_IN:
        # SrcA keeps 19 bits; 32-bit sources must bypass it to keep every bit.
        modes = [ttnn.UnpackToDestMode.Default] * (CB_INPUT_TILES + 1)
        modes[CB_INPUT_TILES] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    return config


def create_program_descriptor(input_tensor, output_tensor, plan):
    tensor_pages = input_tensor.buffer_num_pages()
    assert (
        output_tensor.buffer_num_pages() == tensor_pages
    ), f"output pages {output_tensor.buffer_num_pages()} != input pages {tensor_pages}"

    in_page_bytes = input_tensor.buffer_page_size()
    out_page_bytes = output_tensor.buffer_page_size()
    block_tiles = plan["block_tiles"]
    all_cores = plan["all_cores"]

    cb_input_tiles = ttnn.CBDescriptor(
        total_size=IN_DEPTH * block_tiles * in_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=in_page_bytes
            )
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=OUT_DEPTH * block_tiles * out_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES, data_format=output_tensor.dtype, page_size=out_page_bytes
            )
        ],
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core, first_page, core_tiles in plan["assignment"]:
        reader_rt[core.x][core.y] = [first_page, core_tiles, PAGE_STRIDE]
        writer_rt[core.x][core.y] = [first_page, core_tiles, PAGE_STRIDE]
        compute_rt[core.x][core.y] = [core_tiles]

    reader_ct = [block_tiles, CB_INPUT_TILES, in_page_bytes]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct = [block_tiles, CB_OUTPUT_TILES, out_page_bytes]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=[block_tiles, CB_INPUT_TILES, CB_OUTPUT_TILES],
        runtime_args=compute_rt,
        config=_compute_config(input_tensor.dtype, output_tensor.dtype),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[],
        cbs=[cb_input_tiles, cb_output_tiles],
    )


# Indices of the kernels whose common RT arg 0 is a buffer address (patched on a cache hit).
READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1


def patch_addresses(descriptor, input_tensor, output_tensor):
    descriptor.kernels[READER_KERNEL_INDEX].common_runtime_args[0] = input_tensor.buffer_address()
    descriptor.kernels[WRITER_KERNEL_INDEX].common_runtime_args[0] = output_tensor.buffer_address()
