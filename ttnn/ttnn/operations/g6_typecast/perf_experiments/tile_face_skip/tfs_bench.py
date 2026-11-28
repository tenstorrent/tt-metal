# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tile_face_skip bench: g6_typecast TILE path (regime R1) with reader/writer moving only live faces.

baseline  = the op's own TILE program (g6_typecast_program_descriptor._create_tile_program_descriptor, op kernels
            untouched).
candidates= same work split, same CBs/blocks, same compute kernel + compute config (precision contract identical);
            only the reader / writer kernels (kernels/tfs_*.cpp) and optionally the output CB depth / writer
            flush grouping change.
"""

from __future__ import annotations

from pathlib import Path

import ttnn
from ttnn.operations.g6_typecast import g6_typecast_program_descriptor as pd

HERE = Path(__file__).parent
TFS_KERNELS = HERE / "kernels"

# name -> (reader_face_skip or None (=op reader), writer_face_skip or None (=op writer), out_cb_depth_blocks, write_group)
VARIANTS = {
    "baseline": dict(reader=None, writer=None, out_depth=None, write_group=1),
    "fs_w": dict(reader=None, writer=1, out_depth=None, write_group=1),
    "fs_rw": dict(reader=1, writer=1, out_depth=None, write_group=1),
    "fs_rw_d4": dict(reader=1, writer=1, out_depth=4, write_group=1),
    "fs_rw_g2": dict(reader=1, writer=1, out_depth=4, write_group=2),
}


def _face_geom(dtype, page_bytes):
    if dtype == ttnn.bfloat8_b:
        return 64, 256
    return 0, page_bytes // 4


def build(input_tensor, output_tensor, variant):
    if variant == "baseline":
        return pd._create_tile_program_descriptor(input_tensor, output_tensor)[0]
    v = VARIANTS[variant]
    device = input_tensor.device()
    logical = tuple(input_tensor.shape)
    out_dtype = output_tensor.dtype
    in_page = input_tensor.buffer_page_size()
    out_page = output_tensor.buffer_page_size()
    tensor_tiles = pd._tensor_tiles(logical)
    grid = device.compute_with_storage_grid_size()
    (_, all_cores, g1, g2, t1, t2) = ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)
    out_l1 = 0
    if output_tensor.memory_config().buffer_type == ttnn.BufferType.L1:
        out_l1 = pd._div_up(tensor_tiles, pd._num_l1_banks(device)) * out_page
    block_tiles, depth = pd.plan_blocks(t1, in_page, out_page, out_l1)
    out_depth = depth if v["out_depth"] is None else max(depth, min(v["out_depth"], pd._div_up(t1, block_tiles)))
    write_group = min(v["write_group"], out_depth)

    w = logical[-1]
    h = logical[-2] if len(logical) >= 2 else 1
    Wt, Ht = pd._div_up(w, 32), pd._div_up(h, 32)
    w_tail, h_tail = w % 32, h % 32
    mask = int(out_dtype == ttnn.bfloat8_b and w % 16 != 0)

    cbs = [
        ttnn.CBDescriptor(
            total_size=depth * block_tiles * in_page,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=pd.CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=in_page
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=out_depth * block_tiles * out_page,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=pd.CB_OUTPUT_TILES, data_format=out_dtype, page_size=out_page)
            ],
        ),
    ]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for group, count in ((g1, t1), (g2, t2)):
        if count == 0 or group.num_cores() == 0:
            continue
        for c in ttnn.corerange_to_cores(group, row_wise=True):
            rd_rt[c.x][c.y] = [start, count]
            wr_rt[c.x][c.y] = [start, count]
            cp_rt[c.x][c.y] = [count]
            start += count
    assert start == tensor_tiles

    if v["reader"] is None:
        rd_src = pd.KERNEL_DIR / "g6_typecast_reader.cpp"
        rd_ct = [pd.CB_INPUT_TILES, block_tiles, in_page, mask, Wt, w_tail]
        rd_ct += [0, pd.CB_INPUT_STAGING, 0, 0, 0, pd.STAGING_ALIGN, 1, 1, depth]
    else:
        rd_src = TFS_KERNELS / "tfs_reader.cpp"
        ih, ifc = _face_geom(input_tensor.dtype, in_page)
        rd_ct = [pd.CB_INPUT_TILES, block_tiles, in_page, mask, Wt, w_tail, Ht, h_tail, v["reader"], ih, ifc]
    rd_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    if v["writer"] is None:
        wr_src = pd.KERNEL_DIR / "g6_typecast_writer.cpp"
        wr_ct = [pd.CB_OUTPUT_TILES, block_tiles, out_page, 0, pd.CB_OUTPUT_STAGING, 0, 0, 0, pd.STAGING_ALIGN, 1]
    else:
        wr_src = TFS_KERNELS / "tfs_writer.cpp"
        oh, ofc = _face_geom(out_dtype, out_page)
        wr_ct = [pd.CB_OUTPUT_TILES, block_tiles, out_page, Wt, w_tail, Ht, h_tail, v["writer"], oh, ofc, write_group]
    wr_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(rd_src),
            core_ranges=all_cores,
            compile_time_args=rd_ct,
            runtime_args=rd_rt,
            common_runtime_args=[input_tensor.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(wr_src),
            core_ranges=all_cores,
            compile_time_args=wr_ct,
            runtime_args=wr_rt,
            common_runtime_args=[output_tensor.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(pd.KERNEL_DIR / "g6_typecast_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=[pd.CB_INPUT_TILES, pd.CB_OUTPUT_TILES, block_tiles],
            runtime_args=cp_rt,
            config=pd._compute_config(input_tensor.dtype, out_dtype),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
