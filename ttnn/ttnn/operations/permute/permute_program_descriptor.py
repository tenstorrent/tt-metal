# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for permute — `whole_tile_relocation` regime.

One CB (`cb_tiles`), reader on NoC0, writer on NoC1, one dispatch.

Block knobs (single source of truth — every dependent quantity derives from
these; never restate them as a literal anywhere else):

    BLOCK_TILES  — tile pages per block (transactions per NoC barrier)
    BUFFER_DEPTH — CB depth in blocks (double buffering)
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# --- Block knobs (op_design.md §Blocking Model / §Buffer-depth knobs) --------
BLOCK_TILES = 8  # extent knob: tile pages per block, per barrier
BUFFER_DEPTH = 2  # depth knob: blocks resident in cb_tiles

# Outer-axis block extents. Phase 0: one plane at a time (an outer axis is not
# contiguous with the next in linear tile order once permuted).
BLOCK_N = 1
BLOCK_C = 1

CB_TILES = 0  # reader -> writer


def _div_up(a, b):
    return (a + b - 1) // b


def _grid_assignment(device, num_tiles):
    """Linear OUTPUT-tile range per core, row_wise=True (design §Work Distribution)."""
    grid_size = device.compute_with_storage_grid_size()
    (
        _num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_g1,
        units_g2,
    ) = ttnn.split_work_to_cores(grid_size, num_tiles, row_wise=True)

    assignment = []
    start = 0
    for group, per_core in ((core_group_1, units_g1), (core_group_2, units_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == num_tiles, f"work split covered {start} of {num_tiles} tiles"
    return all_cores, assignment


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    dims,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    in_shape = list(input_tensor.shape)
    rank = len(in_shape)

    # --- tile index space (ceil per image, never floor(N*H/32)) -------------
    in_tile_dims = list(in_shape)
    in_tile_dims[-1] = _div_up(in_shape[-1], TILE_DIM)
    in_tile_dims[-2] = _div_up(in_shape[-2], TILE_DIM)

    # input tile strides, innermost first
    in_stride = [1] * rank
    for d in range(rank - 2, -1, -1):
        in_stride[d] = in_stride[d + 1] * in_tile_dims[d + 1]

    out_tile_dims = [in_tile_dims[d] for d in dims]
    # coefficient of each OUTPUT tile coordinate in the INPUT linear tile index
    out_coeff = [in_stride[d] for d in dims]

    tensor_tiles = 1
    for e in out_tile_dims:
        tensor_tiles *= e
    assert tensor_tiles < 2**31, "permute: tile index space exceeds 32-bit page indices"

    # A plane is BLOCK_N x BLOCK_C outer tiles' worth of the contiguity-carrying
    # (ht, wt) pair — Phase 0: one (ht, wt) plane.
    tiles_per_plane = BLOCK_N * BLOCK_C * out_tile_dims[-2] * out_tile_dims[-1]

    page_bytes = input_tensor.buffer_aligned_page_size()
    assert page_bytes == output_tensor.buffer_aligned_page_size()
    assert input_tensor.buffer_num_pages() == tensor_tiles
    assert output_tensor.buffer_num_pages() == tensor_tiles

    all_cores, assignment = _grid_assignment(device, tensor_tiles)

    cb_tiles = ttnn.CBDescriptor(
        total_size=BUFFER_DEPTH * BLOCK_TILES * page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_TILES,
                data_format=input_tensor.dtype,
                page_size=page_bytes,
            )
        ],
    )

    # CT args: knobs + geometry; TensorAccessorArgs LAST.
    common_ct = [BLOCK_TILES, page_bytes, tiles_per_plane, rank] + out_tile_dims + out_coeff

    reader_ct_args = list(common_ct)
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = list(common_ct)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, start_tile, tiles_this_core in assignment:
        reader_rt_args[core.x][core.y] = [in_addr, start_tile, tiles_this_core]
        writer_rt_args[core.x][core.y] = [out_addr, start_tile, tiles_this_core]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "permute_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "permute_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # NoC1
    )

    # No compute kernel in Phase 0 — pure relocation (see kernels/permute_compute.cpp).
    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb_tiles])
