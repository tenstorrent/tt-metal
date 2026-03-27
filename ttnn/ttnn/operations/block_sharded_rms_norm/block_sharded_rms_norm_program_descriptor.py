# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import struct

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


def _get_rect(core_grid: ttnn.CoreRangeSet) -> ttnn.CoreRange:
    ranges = list(core_grid.ranges())
    if len(ranges) != 1:
        raise ValueError("block_sharded_rms_norm currently requires a single rectangular CoreRangeSet")
    return ranges[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    epsilon: float,
) -> ttnn.ProgramDescriptor:
    shard_spec = input_tensor.memory_config().shard_spec
    core_grid = shard_spec.grid
    core_rect = _get_rect(core_grid)

    shard_h, shard_w = shard_spec.shape
    shard_h_tiles = shard_h // 32
    shard_w_tiles = shard_w // 32
    num_cols = core_rect.end.x - core_rect.start.x + 1
    num_rows = core_rect.end.y - core_rect.start.y + 1
    total_shard_tiles = shard_h_tiles * shard_w_tiles
    total_width = shard_w
    if num_cols > 1:
        total_width *= num_cols

    tile_size = ttnn.tile_size(input_tensor.dtype)
    input_cb = ttnn.cb_descriptor_from_sharded_tensor(0, input_tensor, total_size=total_shard_tiles * tile_size)
    output_cb = ttnn.cb_descriptor_from_sharded_tensor(16, output_tensor, total_size=total_shard_tiles * tile_size)

    def cb_desc(cb_index: int, num_pages: int) -> ttnn.CBDescriptor:
        return ttnn.CBDescriptor(
            total_size=num_pages * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_index,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )

    cb_sq = cb_desc(24, shard_w_tiles)
    cb_partial = cb_desc(25, 1)
    cb_remote_partials = cb_desc(26, num_cols)
    cb_unit_scaler = cb_desc(27, 1)
    cb_inv_rms = cb_desc(28, 1)
    cb_mean_scaler = cb_desc(29, 1)
    cb_eps = cb_desc(30, 1)
    cb_tmp = cb_desc(31, 1)

    sem_id = 0
    semaphores = [ttnn.SemaphoreDescriptor(sem_id, ttnn.CoreType.WORKER, core_grid, 0)]

    epsilon_bits = struct.unpack("<I", struct.pack("<f", float(epsilon)))[0]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=[epsilon_bits],
        runtime_args=_reader_runtime_args(
            input_tensor.device(),
            core_rect,
            total_shard_tiles,
            shard_h_tiles,
            shard_w_tiles,
            num_cols,
            sem_id,
            total_width,
        ),
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=[
            0,  # cb_input
            16,  # cb_output
            24,  # cb_sq
            25,  # cb_partial
            26,  # cb_remote_partials
            27,  # cb_unit_scaler
            28,  # cb_inv_rms
            29,  # cb_mean_scaler
            30,  # cb_eps
            31,  # cb_tmp
            shard_h_tiles,
            shard_w_tiles,
            num_cols,
        ],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=[
            input_cb,
            output_cb,
            cb_sq,
            cb_partial,
            cb_remote_partials,
            cb_unit_scaler,
            cb_inv_rms,
            cb_mean_scaler,
            cb_eps,
            cb_tmp,
        ],
    )


def _reader_runtime_args(
    device: ttnn.Device,
    core_rect: ttnn.CoreRange,
    total_shard_tiles: int,
    shard_h_tiles: int,
    shard_w_tiles: int,
    num_cols: int,
    sem_id: int,
    total_width: int,
) -> ttnn.RuntimeArgs:
    runtime_args = ttnn.RuntimeArgs()

    for y in range(core_rect.start.y, core_rect.end.y + 1):
        physical_row = [
            device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            for x in range(core_rect.start.x, core_rect.end.x + 1)
        ]
        for x in range(core_rect.start.x, core_rect.end.x + 1):
            args = [
                total_shard_tiles,
                shard_h_tiles,
                shard_w_tiles,
                x - core_rect.start.x,
                num_cols,
                sem_id,
                total_width,
            ]
            for physical in physical_row:
                args.extend([physical.x, physical.y])
            runtime_args[x][y] = args

    return runtime_args
