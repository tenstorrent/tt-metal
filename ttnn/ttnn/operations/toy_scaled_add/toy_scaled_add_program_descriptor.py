# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptors for toy_scaled_add: out = a + alpha * (b * gamma).

Two programs, picked by how a / b / out are placed:

  interleaved     a, b, out interleaved (DRAM or L1). The tile-rows are split over the worker grid;
                  readers stream a and b, writers stream out, both through TensorAccessors.
  height_sharded  a, b, out height-sharded on one grid with one shard spec. Each core's a / b / out
                  circular buffers are backed by its own shards, so no tensor data crosses the NoC.

gamma, when given, is interleaved and holds one tile-row ([..., 1, W]); every core loads it once.

Arguments follow kernels/toy_scaled_add_args.hpp: what changes between calls (buffer addresses, alpha)
is in common runtime args, the work split is in per-core runtime args, and gamma's tensor accessor
args are always passed (a placeholder without gamma) so later compile-time offsets never move.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# kernels/toy_scaled_add_args.hpp
CB_A = 0
CB_B = 1
CB_OUT = 2
CB_GAMMA = 3
STREAM_CB_TILES = 2  # double-buffered streaming CBs on the interleaved program

GAMMA_DEFINE = ("TOY_SCALED_ADD_HAS_GAMMA", "1")


def alpha_bits(alpha: float) -> int:
    """alpha as the bit pattern of an fp32, the form the compute kernel's SFPU multiply takes."""
    return struct.unpack("<I", struct.pack("<f", alpha))[0]


def tile_bytes(tensor: ttnn.Tensor) -> int:
    return tensor.tile.get_tile_size(tensor.dtype)


def tile_grid(tensor: ttnn.Tensor) -> tuple[int, int]:
    """(tile-rows, tiles per row) of a tiled tensor, all leading dims folded into the rows."""
    padded = list(tensor.padded_shape)
    tile_h, tile_w = tensor.tile.tile_shape
    width_tiles = padded[-1] // tile_w
    rows = math.prod(padded) // (tile_h * padded[-1])
    return rows, width_tiles


def _tile_format(buffer_index: int, tensor: ttnn.Tensor) -> ttnn.CBFormatDescriptor:
    return ttnn.CBFormatDescriptor(
        buffer_index=buffer_index,
        data_format=tensor.dtype,
        page_size=tile_bytes(tensor),
        tile=ttnn.TileDescriptor(*tensor.tile.tile_shape),
    )


def compute_config(compute_kernel_config, *tensors: ttnn.Tensor) -> ttnn.ComputeConfigDescriptor:
    """The compute kernel's math settings. Without a caller config: HiFi4, exact math, and fp32 DEST
    accumulation whenever an input or the output is fp32."""
    any_fp32 = any(t is not None and t.dtype == ttnn.float32 for t in tensors)
    if compute_kernel_config is None:
        return ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=any_fp32,
        )
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        math_approx_mode=compute_kernel_config.math_approx_mode,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        dst_full_sync_en=compute_kernel_config.dst_full_sync_en,
    )


def _accessor_args(tensor) -> list[int]:
    """Tensor accessor compile-time args; a placeholder of the same layout when the tensor is absent."""
    accessor = ttnn.TensorAccessorArgs(tensor) if tensor is not None else ttnn.TensorAccessorArgs()
    return list(accessor.get_compile_time_args())


def _stream_cb(buffer_index: int, tensor: ttnn.Tensor, num_tiles: int, cores) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=num_tiles * tile_bytes(tensor),
        core_ranges=cores,
        format_descriptors=[_tile_format(buffer_index, tensor)],
    )


def _shard_cb(buffer_index: int, tensor: ttnn.Tensor, cores) -> ttnn.CBDescriptor:
    cb = ttnn.cb_descriptor_from_sharded_tensor(buffer_index, tensor, core_ranges=cores)
    cb.format_descriptors = [_tile_format(buffer_index, tensor)]
    return cb


def _compute_kernel(cores, width_tiles, defines, compute_rt_args, alpha, config) -> ttnn.KernelDescriptor:
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=cores,
        named_compile_time_args=[("Wt", width_tiles)],
        defines=defines,
        runtime_args=compute_rt_args,
        common_runtime_args=[alpha_bits(alpha)],
        config=config,
    )


def create_interleaved_descriptor(a, b, gamma, output, alpha, compute_kernel_config) -> ttnn.ProgramDescriptor:
    num_rows, width_tiles = tile_grid(a)
    grid = a.device().compute_with_storage_grid_size()
    _, all_cores, group_1, group_2, rows_1, rows_2 = ttnn.split_work_to_cores(grid, num_rows, row_wise=True)

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    row_start = 0
    for group, rows in ((group_1, rows_1), (group_2, rows_2)):
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            args = [row_start, rows]
            reader_rt_args[core.x][core.y] = args
            writer_rt_args[core.x][core.y] = args
            compute_rt_args[core.x][core.y] = args
            row_start += rows

    defines = [GAMMA_DEFINE] if gamma is not None else []
    reader_ct_args = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    reader_ct_args += ttnn.TensorAccessorArgs(b).get_compile_time_args()
    reader_ct_args += _accessor_args(gamma)

    cbs = [
        _stream_cb(CB_A, a, STREAM_CB_TILES, all_cores),
        _stream_cb(CB_B, b, STREAM_CB_TILES, all_cores),
        _stream_cb(CB_OUT, output, STREAM_CB_TILES, all_cores),
    ]
    if gamma is not None:
        cbs.append(_stream_cb(CB_GAMMA, gamma, width_tiles, all_cores))

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_interleaved.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        named_compile_time_args=[("Wt", width_tiles)],
        defines=defines,
        runtime_args=reader_rt_args,
        common_runtime_args=[
            a.buffer_address(),
            b.buffer_address(),
            gamma.buffer_address() if gamma is not None else 0,
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_interleaved.cpp"),
        core_ranges=all_cores,
        compile_time_args=list(ttnn.TensorAccessorArgs(output).get_compile_time_args()),
        named_compile_time_args=[("Wt", width_tiles)],
        runtime_args=writer_rt_args,
        common_runtime_args=[output.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = _compute_kernel(
        all_cores,
        width_tiles,
        defines,
        compute_rt_args,
        alpha,
        compute_config(compute_kernel_config, a, b, gamma, output),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def create_height_sharded_descriptor(a, b, gamma, output, alpha, compute_kernel_config) -> ttnn.ProgramDescriptor:
    num_rows, width_tiles = tile_grid(a)
    shard_spec = a.memory_config().shard_spec
    shard_cores = shard_spec.grid
    shard_rows = shard_spec.shape[0] // TILE
    row_wise = shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    # Shard i covers tile-rows [i * shard_rows, (i + 1) * shard_rows); the last one may be partial
    # and trailing cores of the grid may hold none.
    for i, core in enumerate(ttnn.corerange_to_cores(shard_cores, row_wise=row_wise)):
        row_start = min(i * shard_rows, num_rows)
        args = [row_start, min(shard_rows, num_rows - row_start)]
        reader_rt_args[core.x][core.y] = args
        writer_rt_args[core.x][core.y] = args
        compute_rt_args[core.x][core.y] = args

    defines = [GAMMA_DEFINE] if gamma is not None else []
    cbs = [
        _shard_cb(CB_A, a, shard_cores),
        _shard_cb(CB_B, b, shard_cores),
        _shard_cb(CB_OUT, output, shard_cores),
    ]
    if gamma is not None:
        cbs.append(_stream_cb(CB_GAMMA, gamma, width_tiles, shard_cores))

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_sharded.cpp"),
        core_ranges=shard_cores,
        compile_time_args=_accessor_args(gamma),
        named_compile_time_args=[("Wt", width_tiles)],
        defines=defines,
        runtime_args=reader_rt_args,
        common_runtime_args=[gamma.buffer_address() if gamma is not None else 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_sharded.cpp"),
        core_ranges=shard_cores,
        named_compile_time_args=[("Wt", width_tiles)],
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = _compute_kernel(
        shard_cores,
        width_tiles,
        defines,
        compute_rt_args,
        alpha,
        compute_config(compute_kernel_config, a, b, gamma, output),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
