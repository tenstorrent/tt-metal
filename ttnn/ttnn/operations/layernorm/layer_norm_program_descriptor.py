# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import math
import struct

import ttnn


_TILE_WIDTH = 32
_TILE_HEIGHT = 32
_REPO_ROOT = Path(__file__).resolve().parents[4]
_KERNEL_ROOT = _REPO_ROOT / "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels"
_DATAFLOW_KERNEL_DIR = _KERNEL_ROOT / "dataflow"
_COMPUTE_KERNEL_DIR = _KERNEL_ROOT / "compute"
_STAGE_MODE = {
    "data_pipeline": 0,
    "mean_reduce": 1,
    "invstd_reduce": 2,
    "normalize": 3,
    "residual_affine": 4,
    "acceptance": 5,
}


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    epsilon: float,
    stage_name: str,
    residual_input_tensor: ttnn.Tensor | None = None,
    weight: ttnn.Tensor | None = None,
    bias: ttnn.Tensor | None = None,
    is_rmsnorm: bool = False,
) -> ttnn.ProgramDescriptor:
    logical_shape = [int(dim) for dim in input_tensor.shape]
    padded_shape = [int(dim) for dim in input_tensor.padded_shape]
    logical_width = logical_shape[-1]
    padded_width = padded_shape[-1]
    padded_height = padded_shape[-2]
    width_tiles = padded_width // _TILE_WIDTH
    output_width_tiles = int(output_tensor.padded_shape[-1]) // _TILE_WIDTH
    height_tiles = padded_height // _TILE_HEIGHT
    outer_volume = math.prod(padded_shape[:-2]) if len(padded_shape) > 2 else 1
    num_row_units = outer_volume * height_tiles

    device = input_tensor.device()
    compute_grid_size = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))]
    )
    (_, core_grid, core_group_1, core_group_2, rows_per_core_group_1, rows_per_core_group_2) = ttnn.split_work_to_cores(
        all_cores, num_row_units
    )

    intermediate_dtype = ttnn.float32 if output_tensor.dtype == ttnn.float32 else input_tensor.dtype
    fp32_dest_acc_en = intermediate_dtype == ttnn.float32

    reader_ct_args = [int(residual_input_tensor is not None), int(weight is not None), int(bias is not None)]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(residual_input_tensor).get_compile_time_args()
        if residual_input_tensor is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(weight).get_compile_time_args()
        if weight is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(bias).get_compile_time_args()
        if bias is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    stage_mode = _STAGE_MODE[stage_name]

    compute_ct_args = [
        int(residual_input_tensor is not None),
        int(weight is not None),
        int(bias is not None),
        int(is_rmsnorm),
        stage_mode,
    ]
    writer_ct_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    input_addr = input_tensor.buffer_address()
    residual_addr = residual_input_tensor.buffer_address() if residual_input_tensor is not None else 0
    weight_addr = weight.buffer_address() if weight is not None else 0
    bias_addr = bias.buffer_address() if bias is not None else 0
    output_addr = output_tensor.buffer_address()
    inv_width_bf16_packed = _float_to_bfloat16_packed(1.0 / float(logical_width))
    epsilon_bits = _float_to_uint32(epsilon)

    reader_runtime_prefix = [input_addr, residual_addr, weight_addr, bias_addr]
    reader_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    _initialize_runtime_args(reader_rt_args, core_grid)
    _initialize_runtime_args(compute_rt_args, core_grid)
    _initialize_runtime_args(writer_rt_args, core_grid)

    tile_offset = 0
    tile_offset = _assign_runtime_args(
        core_group_1,
        rows_per_core_group_1,
        tile_offset,
        width_tiles,
        output_width_tiles,
        logical_width,
        reader_runtime_prefix,
        inv_width_bf16_packed,
        epsilon_bits,
        stage_mode,
        output_addr,
        reader_rt_args,
        compute_rt_args,
        writer_rt_args,
    )
    _assign_runtime_args(
        core_group_2,
        rows_per_core_group_2,
        tile_offset,
        width_tiles,
        output_width_tiles,
        logical_width,
        reader_runtime_prefix,
        inv_width_bf16_packed,
        epsilon_bits,
        stage_mode,
        output_addr,
        reader_rt_args,
        compute_rt_args,
        writer_rt_args,
    )

    cbs = [
        _make_cb_descriptor(0, input_tensor.dtype, 2, core_grid),
        _make_cb_descriptor(1, input_tensor.dtype, 2, core_grid),
        _make_cb_descriptor(2, input_tensor.dtype, 2, core_grid),
        _make_cb_descriptor(3, input_tensor.dtype, 2, core_grid),
        _make_cb_descriptor(4, input_tensor.dtype, 2, core_grid),
        _make_cb_descriptor(5, input_tensor.dtype, 2, core_grid),
        _make_cb_descriptor(6, ttnn.bfloat16, 1, core_grid),
        _make_cb_descriptor(7, intermediate_dtype, 1, core_grid),
        _make_cb_descriptor(8, input_tensor.dtype, 1, core_grid),
        _make_cb_descriptor(16, output_tensor.dtype, 2, core_grid),
        _make_cb_descriptor(24, intermediate_dtype, 1, core_grid),
        _make_cb_descriptor(25, intermediate_dtype, 1, core_grid),
        _make_cb_descriptor(26, intermediate_dtype, 1, core_grid),
        _make_cb_descriptor(27, intermediate_dtype, 1, core_grid),
        _make_cb_descriptor(28, intermediate_dtype, 1, core_grid),
        _make_cb_descriptor(29, intermediate_dtype, 1, core_grid),
        _make_cb_descriptor(30, intermediate_dtype, 1, core_grid),
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(_DATAFLOW_KERNEL_DIR / "reader_layernorm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(_DATAFLOW_KERNEL_DIR / "writer_layernorm.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(
            _COMPUTE_KERNEL_DIR / ("layernorm_sfpu_kernel.cpp" if fp32_dest_acc_en else "layernorm_kernel.cpp")
        ),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )


def _initialize_runtime_args(runtime_args: ttnn.RuntimeArgs, core_ranges: ttnn.CoreRangeSet) -> None:
    for core_range in core_ranges.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                runtime_args[x][y] = []


def _assign_runtime_args(
    core_group: ttnn.CoreRangeSet,
    rows_per_core: int,
    tile_offset: int,
    width_tiles: int,
    output_width_tiles: int,
    logical_width: int,
    reader_runtime_prefix: list[int],
    inv_width_bf16_packed: int,
    epsilon_bits: int,
    stage_mode: int,
    output_addr: int,
    reader_rt_args: ttnn.RuntimeArgs,
    compute_rt_args: ttnn.RuntimeArgs,
    writer_rt_args: ttnn.RuntimeArgs,
) -> int:
    if rows_per_core == 0:
        return tile_offset

    for core_range in core_group.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [
                    *reader_runtime_prefix,
                    rows_per_core,
                    width_tiles,
                    tile_offset,
                    logical_width,
                    inv_width_bf16_packed,
                    epsilon_bits,
                    stage_mode,
                ]
                compute_rt_args[x][y] = [rows_per_core, width_tiles, logical_width]
                writer_rt_args[x][y] = [output_addr, rows_per_core, width_tiles, output_width_tiles, tile_offset]
                tile_offset += rows_per_core * width_tiles
    return tile_offset


def _make_cb_descriptor(
    buffer_index: int,
    dtype: ttnn.DataType,
    num_pages: int,
    core_grid: ttnn.CoreRangeSet,
) -> ttnn.CBDescriptor:
    page_size = ttnn.tile_size(dtype)
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=buffer_index,
                data_format=dtype,
                page_size=page_size,
            )
        ],
    )


def _float_to_bfloat16_packed(value: float) -> int:
    float_bytes = struct.pack("f", float(value))
    bf16_bytes = float_bytes[2:4]
    return int.from_bytes(bf16_bytes + bf16_bytes, byteorder="little")


def _float_to_uint32(value: float) -> int:
    return int.from_bytes(struct.pack("f", float(value)), byteorder="little")
