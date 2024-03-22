# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Union

import ttnn

import tt_lib as ttl
from tt_lib.utils import find_closest_largest_divisor, find_closest_largest_divisor_with_num_padding
import math


def _torch_layer_norm(
    input_tensor: ttnn.Tensor, *, epsilon=1e-12, residual_input_tensor=None, weight=None, bias=None, **_
):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    if residual_input_tensor is not None:
        residual_input_tensor = ttnn.from_device(residual_input_tensor)
        residual_input_tensor = ttnn.to_layout(residual_input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        residual_input_tensor = ttnn.to_torch(residual_input_tensor)
        input_tensor += residual_input_tensor

    if weight is not None:
        weight = ttnn.from_device(weight)
        weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        weight = ttnn.to_torch(weight)
        if len(weight.shape) == 2:
            weight = weight[0]

    if bias is not None:
        bias = ttnn.from_device(bias)
        bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.to_torch(bias)
        if len(bias.shape) == 2:
            bias = bias[0]

    return torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight, bias, eps=epsilon)


def _layer_norm_validate_input_tensors(
    operation_name, input_tensor, *args, weight=None, bias=None, residual_input_tensor=None, **kwargs
):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        weight,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )
    ttnn.validate_input_tensor(
        operation_name,
        bias,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )
    ttnn.validate_input_tensor(
        operation_name,
        residual_input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )


@ttnn.register_operation(
    name="ttnn.layer_norm",
    validate_input_tensors=_layer_norm_validate_input_tensors,
    torch_function=_torch_layer_norm,
)
def layer_norm(
    input_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-12,
    weight: Optional[ttnn.Tensor] = None,
    bias: Optional[ttnn.Tensor] = None,
    residual_input_tensor: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    program_config: Optional[ttnn.experimental.operations.primary.LayerNormShardedMultiCoreProgramConfig] = None,
) -> ttnn.Tensor:
    r"""
    layer_norm(input_tensor: ttnn.Tensor, *, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None, residual_input_tensor: Optional[ttnn.Tensor] = None, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Compute layer_norm over :attr:`input_tensor`.

    """

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    if residual_input_tensor is not None:
        residual_input_tensor = ttnn.unsqueeze_to_4D(residual_input_tensor)
    if weight is not None:
        weight = ttnn.unsqueeze_to_4D(weight)
    if bias is not None:
        bias = ttnn.unsqueeze_to_4D(bias)

    if program_config is None:
        if residual_input_tensor is not None:
            output_tensor = ttnn.experimental.tensor.add_layernorm(
                input_tensor, residual_input_tensor, epsilon, weight, bias, output_mem_config=memory_config
            )
        else:
            output_tensor = ttnn.experimental.tensor.layernorm(
                input_tensor,
                epsilon,
                weight,
                bias,
                output_mem_config=memory_config,
            )
    else:
        if residual_input_tensor is not None:
            output_tensor = ttnn.experimental.operations.primary.add_layernorm(
                input_tensor,
                residual_input_tensor,
                epsilon,
                weight,
                bias,
                output_mem_config=memory_config,
                program_config=program_config,
            )
        else:
            output_tensor = ttnn.experimental.operations.primary.layernorm(
                input_tensor, epsilon, weight, bias, output_mem_config=memory_config, program_config=program_config
            )

    output_tensor = ttnn.reshape(output_tensor, original_shape)
    return output_tensor


def _rms_norm_validate_input_tensors(operation_name, input_tensor, weight, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        weight,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.rms_norm",
    validate_input_tensors=_rms_norm_validate_input_tensors,
)
def rms_norm(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, epsilon: float = 1e-6) -> ttnn.Tensor:
    r"""
    rms_norm(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, epsilon: float = 1e-6) -> ttnn.Tensor

    Compute rms_norm over :attr:`input_tensor`.

    """

    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise RuntimeError("rms_norm only supports device storage type")

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    weight = ttnn.unsqueeze_to_4D(weight)

    output_tensor = ttl.tensor.rmsnorm(input_tensor, epsilon, weight)

    output_tensor = ttnn.reshape(output_tensor, original_shape)

    return output_tensor


# group norm helper function
def determine_expected_group_norm_sharded_config_and_grid_size(
    *, device, num_channels, num_groups, input_nhw, is_height_sharded
):
    assert num_channels % num_groups == 0
    assert num_channels % 32 == 0  # TODO: remove this later
    group_size = num_channels // num_groups
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = (compute_with_storage_grid_size.x, compute_with_storage_grid_size.y)
    max_num_cores = device_grid_size[0] * device_grid_size[1]
    input_nhw_paddedto32 = math.ceil(input_nhw / 32) * 32
    num_cores_nhw = find_closest_largest_divisor(
        input_nhw_paddedto32 // 32, max_num_cores if is_height_sharded else device_grid_size[0]
    )
    if is_height_sharded:
        num_cores_channels = 1
    else:
        num_cores_channels = device_grid_size[1]
        # num_channels_tiles = num_channels // 16
        num_channels_tiles = num_channels // 8
        while (num_channels_tiles % num_cores_channels != 0) or (
            ((num_channels // num_cores_channels) % group_size) != 0
        ):
            num_cores_channels -= 1
            assert num_cores_channels > 0
    input_nhw_padded_to_ncores = math.ceil(input_nhw / (num_cores_nhw * 32)) * (num_cores_nhw * 32)
    gn_in_channels_per_core = num_channels // num_cores_channels
    # assert gn_in_channels_per_core % 16 == 0
    assert gn_in_channels_per_core % 8 == 0
    gn_nhw_per_core = input_nhw_padded_to_ncores // num_cores_nhw
    if is_height_sharded:
        grid_size = [
            device_grid_size[0] if num_cores_nhw >= device_grid_size[0] else num_cores_nhw,
            math.ceil(num_cores_nhw / device_grid_size[0]),
        ]  # for 1d systolic array, grid size is the tightest bound of num_cores_nhw as a rectangle (x,y)
        assert (
            num_cores_nhw <= grid_size[0] * grid_size[1]
        ), "Error: For height sharding, num_cores_nhw must be <= grid size"
    else:
        grid_size = [num_cores_nhw, num_cores_channels]
    shard_strategy = ttnn.ShardStrategy.HEIGHT if is_height_sharded else ttnn.ShardStrategy.BLOCK
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR if is_height_sharded else ttnn.ShardOrientation.COLUMN_MAJOR
    return ttnn.create_sharded_memory_config(
        (1, 1, gn_in_channels_per_core, gn_nhw_per_core),
        ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        shard_strategy,
        shard_orientation,
        halo=False,
        use_height_and_width_as_shard_shape=True,
    ), ttnn.CoreGrid(y=grid_size[1], x=grid_size[0])


# for debug purpose
def manual_group_norm(input_tensor, num_groups, eps=1e-2):
    import torch

    N, C, H, W = input_tensor.shape
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"

    # Reshape into groups
    group_channels = C // num_groups
    input_tensor = input_tensor.view(N, num_groups, group_channels, H, W)

    # Calculate mean and variance
    mean = input_tensor.mean(dim=(2, 3, 4), keepdim=True)
    var = input_tensor.var(dim=(2, 3, 4), keepdim=True)

    # Normalize
    input_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    # Reshape back to original dimensions
    input_tensor = input_tensor.view(N, C, H, W)
    return input_tensor


def create_group_norm_weight_bias_rm(input_tensor, num_channels, num_groups):
    import torch

    def find_ceil_divisible_by_32(n):
        return ((n + 31) // 32) * 32

    values_per_chunk = num_channels // num_groups
    zeros_to_insert = find_ceil_divisible_by_32(values_per_chunk) - values_per_chunk
    input_tensor = input_tensor.view(-1, values_per_chunk)
    input_tensor = torch.nn.functional.pad(input_tensor, (0, zeros_to_insert))
    input_tensor = input_tensor.flatten()
    input_tensor = input_tensor[: num_channels + zeros_to_insert * (num_channels // values_per_chunk)]
    return input_tensor.reshape(1, 1, -1, 32)


def find_max_tile_span(W, group_size, tile_width):
    current_position = 0
    max_tile_span = 0

    while current_position < W:
        group_end = current_position + group_size
        start_tile = current_position // tile_width
        end_tile = (group_end - 1) // tile_width
        current_tile_span = end_tile - start_tile + 1
        max_tile_span = max(max_tile_span, current_tile_span)
        current_position = group_end
    return max_tile_span


def create_groupnorm_input_mask(num_channel, num_groups, num_cores_across_channel):
    import torch

    block_wt = find_max_tile_span(num_channel, num_channel // num_groups, 32)
    input_mask_tensor = torch.zeros((1, num_groups, 32, int(32 * block_wt)), dtype=torch.bfloat16)

    num_groups_per_core = num_groups // num_cores_across_channel
    num_cols_per_group = num_channel // num_groups

    start_strides = []
    for _ in range(num_cores_across_channel):
        row_offset = 0
        start_strides.append(0)
        for _ in range(num_groups_per_core - 1):
            if row_offset + (num_cols_per_group % 32) == 32:
                row_offset = 0
            elif row_offset + (num_cols_per_group % 32) > 32:
                row_offset = (num_cols_per_group % 32) + row_offset - 32
            else:
                row_offset += num_cols_per_group % 32
            start_strides.append(row_offset)
        end_strides = [i + num_cols_per_group for i in start_strides]

    for group in range(num_groups):
        start_stride = start_strides[group]
        end_stride = end_strides[group]
        end_stride = min(end_stride, input_mask_tensor.shape[3])
        input_mask_tensor[:, group, :, start_stride:end_stride] = 1

    return input_mask_tensor


def _torch_group_norm(input_tensor: ttnn.Tensor, *, num_groups, epsilon=1e-05, weight=None, bias=None, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    if weight is not None:
        weight = ttnn.from_device(weight)
        weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        weight = ttnn.to_torch(weight)
        if len(weight.shape) == 2:
            weight = weight[0]

    if bias is not None:
        bias = ttnn.from_device(bias)
        bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.to_torch(bias)
        if len(bias.shape) == 2:
            bias = bias[0]
    return torch.nn.functional.group_norm(input_tensor, num_groups, weight, bias, eps=epsilon)


def _fallback_group_norm(input_tensor: ttnn.Tensor, *, num_groups, epsilon=1e-05, weight=None, bias=None, **_):
    output_tensor = _torch_group_norm(input_tensor, num_groups=num_groups, epsilon=epsilon, weight=weight, bias=bias)
    output_tensor = ttnn.from_torch(
        output_tensor, dtype=input_tensor.dtype, layout=input_tensor.layout, device=input_tensor.device()
    )
    output_tensor = ttnn.reshape(output_tensor, input_tensor.shape)
    return output_tensor


def _group_norm_validate_input_tensors(operation_name, input_tensor, *args, weight=None, bias=None, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16,),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        weight,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16,),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )
    ttnn.validate_input_tensor(
        operation_name,
        bias,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16,),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
    )


@ttnn.register_operation(
    name="ttnn.group_norm",
    validate_input_tensors=_group_norm_validate_input_tensors,
    torch_function=_torch_group_norm,
    fallback=_fallback_group_norm,
)
def group_norm(
    input_tensor: ttnn.Tensor,
    *,
    num_groups: int,
    epsilon: float = 1e-12,
    input_mask: Optional[ttnn.Tensor] = None,
    weight: Optional[ttnn.Tensor] = None,
    bias: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[Union[ttnn.CoreGrid, ttnn.CoreRange]] = None,
    inplace: Optional[bool] = True,
) -> ttnn.Tensor:
    r"""
    group_norm(input_tensor: ttnn.Tensor, *, num_groups: int, epsilon: float = 1e-12, weight: Optional[ttnn.Tensor] = None, bias: Optional[ttnn.Tensor] = None) -> ttnn.Tensor

    Compute group_norm over :attr:`input_tensor`.

    """

    if core_grid is not None and not isinstance(core_grid, ttnn.CoreGrid):
        raise RuntimeError("core_grid must be a valid CoreGrid object")

    if ttnn.is_sharded(input_tensor):
        if input_tensor.shape.rank != 4:
            raise TypeError("The input tensor rank must equal to 4")

        if input_tensor.shape[-1] % num_groups != 0:
            raise TypeError("number of channels must be divisible by number of groups")

        if ttnn.get_memory_config(input_tensor).memory_layout == ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
            raise TypeError("Cannot be width sharded")

        if (input_tensor.shape[0] * input_tensor.shape[1] * input_tensor.shape[2]) % ttnn.TILE_SIZE != 0:
            raise TypeError("input tensor dim NHW must be divisible by tile size")

        output_dtype = input_tensor.dtype if dtype is None else dtype

        if weight is not None:
            weight = ttnn.unsqueeze_to_4D(weight)

        if bias is not None:
            bias = ttnn.unsqueeze_to_4D(bias)

        output_tensor = ttnn.experimental.operations.primary.groupnorm(
            input_tensor,
            num_groups,
            epsilon,
            weight,
            bias,
            input_mask,
            output_mem_config=memory_config,
            program_config=ttl.operations.primary.GroupNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid.x, core_grid.y),
                out_data_format=output_dtype,
                inplace=inplace,
            ),
        )
        return output_tensor

    else:
        raise NotImplementedError


__all__ = []
