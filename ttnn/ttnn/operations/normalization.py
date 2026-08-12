# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import ttnn

import math

from ttnn._ttnn.operations.normalization import (
    create_group_norm_input_mask,
    create_group_norm_input_negative_mask,
    determine_expected_group_norm_sharded_config_and_grid_size,
    _compute_num_virtual_cols,
    _find_expected_dram_grid,
)


def find_closest_largest_divisor(num: int, start_divisor: int):
    """Return the largest divisor of num that is <= start_divisor.

    Used to choose a core count that divides a work quota. Assumes
    1 <= start_divisor <= num. Decrements until a divisor is found.
    """
    divisor = start_divisor
    while num % divisor != 0:
        divisor = divisor - 1
    return divisor


def _golden_function(input_tensor: ttnn.Tensor, dim: Optional[int] = None, **_):
    import torch

    dim = dim or -1

    return torch.nn.Softmax(dim)(input_tensor)


ttnn.attach_golden_function(
    ttnn.softmax,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.softmax_in_place,
    golden_function=_golden_function,
)


def _golden_function(input_tensor: ttnn.Tensor, scalar: float, attention_mask=None, **_):
    import torch

    input_tensor = input_tensor.float()
    input_tensor = input_tensor * scalar
    if attention_mask is not None:
        input_tensor = input_tensor + attention_mask
    return torch.softmax(input_tensor, dim=-1)


ttnn.attach_golden_function(
    ttnn.scale_mask_softmax_in_place,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.scale_mask_softmax,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.scale_causal_mask_hw_dims_softmax_in_place,
    golden_function=_golden_function,
)


SoftmaxProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxProgramConfig
SoftmaxDefaultProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxDefaultProgramConfig
SoftmaxShardedMultiCoreProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxShardedMultiCoreProgramConfig


def _golden_function(
    input_tensor: ttnn.Tensor,
    *,
    epsilon=1e-12,
    residual_input_tensor=None,
    weight=None,
    bias=None,
    **_,
):
    import torch

    if residual_input_tensor is not None:
        input_tensor += residual_input_tensor

    if weight is not None:
        if len(weight.shape) >= 2:
            weight = weight.squeeze()
        weight = weight.to(input_tensor.dtype)

    if bias is not None:
        if len(bias.shape) >= 2:
            bias = bias.squeeze()
        bias = bias.to(input_tensor.dtype)

    return torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight, bias, eps=epsilon)


ttnn.attach_golden_function(ttnn.layer_norm, golden_function=_golden_function)


def _golden_function(input_tensor: ttnn.Tensor, weight=None, *, epsilon=1e-12, **_):
    import torch

    variance = input_tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
    input_tensor = input_tensor * torch.rsqrt(variance + epsilon)

    if weight is not None and weight.dtype in [torch.float16, torch.bfloat16]:
        input_tensor = input_tensor.to(weight.dtype)

    return weight * input_tensor if weight is not None else input_tensor


ttnn.attach_golden_function(ttnn.rms_norm, golden_function=_golden_function)


def _golden_function_batch_norm(
    input,
    *,
    running_mean=None,
    running_var=None,
    training=False,
    eps=1e-5,
    momentum=0.1,
    weight=None,
    bias=None,
    output=None,
    **_,
):
    import torch

    channels = input.shape[1]

    def channel_vector(tensor):
        return None if tensor is None else tensor.reshape(channels)

    mean = channel_vector(running_mean)
    variance = channel_vector(running_var)
    mean_fp32 = None if mean is None else mean.float()
    variance_fp32 = None if variance is None else variance.float()
    normalized = torch.nn.functional.batch_norm(
        input.float(),
        None if training else mean_fp32,
        None if training else variance_fp32,
        None if weight is None else channel_vector(weight).float(),
        None if bias is None else channel_vector(bias).float(),
        training=training,
        momentum=momentum,
        eps=eps,
    )
    if training:
        reduction_dims = (0, *range(2, input.ndim))
        input_fp32 = input.float()
        batch_mean = input_fp32.mean(dim=reduction_dims)
        batch_variance = (
            (input_fp32 - batch_mean.reshape(1, channels, *([1] * (input.ndim - 2)))).square().mean(dim=reduction_dims)
        )
        if mean is not None:
            updated_mean = (1.0 - momentum) * mean_fp32 + momentum * batch_mean
            mean.copy_(updated_mean.to(mean.dtype))
        if variance is not None:
            # TTNN's running-statistics kernel consumes the same biased variance
            # used for normalization (unlike torch.batch_norm's unbiased update).
            updated_variance = (1.0 - momentum) * variance_fp32 + momentum * batch_variance
            variance.copy_(updated_variance.to(variance.dtype))

    normalized = normalized.to(input.dtype)
    if output is not None:
        if output.shape != normalized.shape or output.dtype != normalized.dtype:
            raise ValueError("batch_norm output must match the input shape and dtype")
        output.copy_(normalized)
        return output
    return normalized


ttnn.attach_golden_function(ttnn.batch_norm, golden_function=_golden_function_batch_norm)


def _distributed_norm_uses_welford(program_config):
    return bool(program_config is not None and getattr(program_config, "use_welford", False))


def _golden_function_norm_pre_all_gather(
    input_tensor,
    *,
    residual_input_tensor=None,
    program_config=None,
    rms_norm=False,
    **_,
):
    import torch

    value = input_tensor if residual_input_tensor is None else input_tensor + residual_input_tensor
    value_fp32 = value.float()
    zeros = torch.zeros(*value.shape[:-1], 31, dtype=value_fp32.dtype, device=value.device)

    if rms_norm:
        sum_square = value_fp32.square().sum(dim=-1, keepdim=True)
        return torch.cat((sum_square, zeros), dim=-1)

    if _distributed_norm_uses_welford(program_config):
        mean = value_fp32.mean(dim=-1, keepdim=True)
        variance = value_fp32.var(dim=-1, keepdim=True, unbiased=False)
        return torch.cat((mean, zeros, variance, zeros), dim=-1)

    sum_square = value_fp32.square().sum(dim=-1, keepdim=True)
    value_sum = value_fp32.sum(dim=-1, keepdim=True)
    return torch.cat((sum_square, zeros, value_sum, zeros), dim=-1)


def _golden_function_layer_norm_pre_all_gather(input_tensor, **kwargs):
    return _golden_function_norm_pre_all_gather(input_tensor, rms_norm=False, **kwargs)


def _golden_function_rms_norm_pre_all_gather(input_tensor, **kwargs):
    return _golden_function_norm_pre_all_gather(input_tensor, rms_norm=True, **kwargs)


ttnn.attach_golden_function(
    ttnn.layer_norm_pre_all_gather,
    golden_function=_golden_function_layer_norm_pre_all_gather,
)
ttnn.attach_golden_function(
    ttnn.rms_norm_pre_all_gather,
    golden_function=_golden_function_rms_norm_pre_all_gather,
)


def _apply_distributed_norm_affine(output, weight, bias):
    if weight is not None:
        output = output * weight.reshape(-1).to(output.dtype)
    if bias is not None:
        output = output + bias.reshape(-1).to(output.dtype)
    return output


def _cast_distributed_norm_output(output, dtype, input_dtype):
    import torch

    if dtype == ttnn.float32:
        return output.to(torch.float32)
    if dtype == ttnn.bfloat16:
        return output.to(torch.bfloat16)
    return output.to(input_dtype)


def _golden_function_layer_norm_post_all_gather(
    input_tensor,
    stats,
    *,
    epsilon=1e-12,
    weight=None,
    bias=None,
    program_config=None,
    dtype=None,
    **_,
):
    import torch

    stats = stats.float()
    if stats.shape[-1] % 64 != 0:
        raise ValueError("layer_norm_post_all_gather expects one 64-column stats block per device")

    blocks = stats.reshape(*stats.shape[:-1], -1, 64)
    if _distributed_norm_uses_welford(program_config):
        local_mean = blocks[..., 0]
        local_variance = blocks[..., 32]
        mean = local_mean.mean(dim=-1, keepdim=True)
        variance = (local_variance + local_mean.square()).mean(dim=-1, keepdim=True) - mean.square()
    else:
        global_width = input_tensor.shape[-1] * blocks.shape[-2]
        sum_square = blocks[..., 0].sum(dim=-1, keepdim=True)
        value_sum = blocks[..., 32].sum(dim=-1, keepdim=True)
        mean = value_sum / global_width
        variance = sum_square / global_width - mean.square()

    output = (input_tensor.float() - mean) * torch.rsqrt(variance.clamp_min(0) + epsilon)
    output = _apply_distributed_norm_affine(output, weight, bias)
    return _cast_distributed_norm_output(output, dtype, input_tensor.dtype)


def _golden_function_rms_norm_post_all_gather(
    input_tensor,
    stats,
    *,
    epsilon=1e-12,
    weight=None,
    bias=None,
    dtype=None,
    **_,
):
    import torch

    stats = stats.float()
    if stats.shape[-1] % 32 != 0:
        raise ValueError("rms_norm_post_all_gather expects one 32-column stats block per device")
    blocks = stats.reshape(*stats.shape[:-1], -1, 32)
    global_width = input_tensor.shape[-1] * blocks.shape[-2]
    mean_square = blocks[..., 0].sum(dim=-1, keepdim=True) / global_width
    output = input_tensor.float() * torch.rsqrt(mean_square + epsilon)
    output = _apply_distributed_norm_affine(output, weight, bias)
    return _cast_distributed_norm_output(output, dtype, input_tensor.dtype)


ttnn.attach_golden_function(
    ttnn.layer_norm_post_all_gather,
    golden_function=_golden_function_layer_norm_post_all_gather,
)
ttnn.attach_golden_function(
    ttnn.rms_norm_post_all_gather,
    golden_function=_golden_function_rms_norm_post_all_gather,
)

# fused_rms_minimal is deliberately left unattached. Its observable contract includes
# mutation of the caller-owned stats/global-CB buffer and a topology-dependent fused
# all-gather. A single CPU tensor cannot represent those device-local state transitions.

LayerNormProgramConfig = ttnn._ttnn.operations.normalization.LayerNormProgramConfig
LayerNormDefaultProgramConfig = ttnn._ttnn.operations.normalization.LayerNormDefaultProgramConfig
LayerNormShardedMultiCoreProgramConfig = ttnn._ttnn.operations.normalization.LayerNormShardedMultiCoreProgramConfig
LayerNormType = ttnn._ttnn.operations.normalization.LayerNormType
DistributedLayerNormStage = ttnn._ttnn.operations.normalization.DistributedLayerNormStage
LayerNormParams = ttnn._ttnn.operations.normalization.LayerNormParams
LayerNormInputs = ttnn._ttnn.operations.normalization.LayerNormInputs
LayerNormDeviceOperation = ttnn._ttnn.operations.normalization.LayerNormDeviceOperation
LayerNormMultiCoreProgramFactory = ttnn._ttnn.operations.normalization.LayerNormMultiCoreProgramFactory
LayerNormShardedProgramFactory = ttnn._ttnn.operations.normalization.LayerNormShardedProgramFactory
layernorm_default_compute_config = ttnn._ttnn.operations.normalization.layernorm_default_compute_config
rmsnorm_default_compute_config = ttnn._ttnn.operations.normalization.rmsnorm_default_compute_config
create_layernorm_program_config = ttnn._ttnn.operations.normalization.create_layernorm_program_config


def create_layer_norm_reciprocals(device: ttnn.Device, core_range_set: ttnn.CoreRangeSet, width: int):
    """
    Create reciprocals tensor for layer norm with Welford algorithm.

    Generates reciprocal values [1/1, 1/2, 1/3, ..., 1/width] where width is
    the per-core width in elements. The tensor is replicated for each core so that
    when sharded to L1 memory, each core has a complete copy.

    This tensor is required when using the Welford algorithm (use_welford=True).

    Args:
        device: The device to create the tensor on.
        core_range_set: The set of cores to shard the reciprocals across.
        width: The width per core in elements (for sharded inputs, this is shard_spec.shape[1];
               for non-sharded inputs, this is the full tensor width).

    Returns:
        A HEIGHT_SHARDED tensor in L1 with shape (num_cores, width) containing
        the reciprocal lookup table values in float32 format.

    Example:
        >>> # For sharded input
        >>> shard_spec = input_tensor.memory_config().shard_spec
        >>> recip_tensor = ttnn.create_layer_norm_reciprocals(
        ...     device, shard_spec.grid, shard_spec.shape[1]
        ... )
        >>> # For non-sharded input
        >>> grid = device.compute_with_storage_grid_size()
        >>> core_range_set = ttnn.CoreRangeSet({
        ...     ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))
        ... })
        >>> recip_tensor = ttnn.create_layer_norm_reciprocals(
        ...     device, core_range_set, input_tensor.shape[-1]
        ... )
    """
    import torch

    num_cores = core_range_set.num_cores()

    # Compute reciprocals: 1/1, 1/2, 1/3, ..., 1/width
    reciprocals = [1.0 / (i + 1) for i in range(width)]

    # Replicate for all cores
    all_reciprocals = reciprocals * num_cores

    # Create torch tensor
    torch_tensor = torch.tensor(all_reciprocals, dtype=torch.float32).reshape(num_cores, width)

    # Create shard spec and memory config for HEIGHT_SHARDED L1
    recip_shard_spec = ttnn.ShardSpec(
        core_range_set,
        (1, width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        recip_shard_spec,
    )

    # Convert to ttnn tensor on device
    recip_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    return recip_tensor


# group norm helper function


def determine_expected_group_norm_dram_grid_size(*, device, num_channels, num_groups, input_nhw, num_batches=1):
    """Determine a valid core grid for DRAM interleaved (non-sharded) group norm.

    Delegates to the C++ implementation which finds the largest grid (x then y)
    within the device compute grid that satisfies the DRAM group-norm constraints.

    Args:
        num_batches: Number of batches (N dimension). Used to ensure uniform
            multicast group sizes for correct kernel synchronization.

    Returns: CoreGrid
    """
    assert num_channels % num_groups == 0
    assert num_channels % ttnn.TILE_SIZE == 0
    compute_grid = device.compute_with_storage_grid_size()
    return _find_expected_dram_grid(compute_grid.x, compute_grid.y, num_channels, num_groups, input_nhw, num_batches)


def create_group_norm_weight_bias_rm(input_tensor, num_channels, num_cores_x):
    """Prepares a gamma/beta tensor in a padded [1,1,-1,32] format.

    - Splits channels into num_cores_x equal chunks
    - Pads each chunk to a multiple of 32 (tile width).
    - Returns a tensor reshaped to [1, 1, tiles_per_core_total, 32].
    """
    import torch

    def find_ceil_divisible_by_32(n):
        return ((n + 31) // 32) * 32

    values_per_chunk = num_channels // num_cores_x
    zeros_to_insert = find_ceil_divisible_by_32(values_per_chunk) - values_per_chunk
    input_tensor = input_tensor.view(-1, values_per_chunk)
    input_tensor = torch.nn.functional.pad(input_tensor, (0, zeros_to_insert))
    input_tensor = input_tensor.flatten()
    input_tensor = input_tensor[: num_channels + zeros_to_insert * (num_channels // values_per_chunk)]
    return input_tensor.reshape(1, 1, -1, 32)


def dram_group_norm_virtual_columns(core_grid, num_channels, num_groups):
    """Choose number of virtual columns for DRAM params/mask generation.

    Delegates to the C++ implementation of compute_num_virtual_cols.
    """
    result = _compute_num_virtual_cols(core_grid.x, num_groups, num_channels)
    assert result > 0, (
        f"dram_group_norm_virtual_columns: could not find a valid num_virtual_cols for "
        f"grid_x={core_grid.x}, num_channels={num_channels}, num_groups={num_groups}"
    )
    return result


def dram_group_norm_params_from_torch(
    torch_params,
    channels_per_device,
    groups_per_device,
    device,
    mesh_axis=None,
    core_grid=None,
    return_mask=True,
    dtype=ttnn.bfloat16,
):
    """
    Create group norm parameters from torch in row major layout. It currently supports sharding along 1 mesh dimension. Sharding along 2 dimensions to be added as needed.
    Args:
        torch_params: List[torch.Tensor] or torch.Tensor. This is weight and or bias for the affine transformation.
        channels_per_device: Number of channels per device if using multi-device else number of channels
        groups_per_device: Number of groups per device if using multi-device else number of groups
        device: Device to create the group norm parameters on. Set to None if setting up on host. Must be provided if core_grid is None
        mesh_axis: Axis to shard the parameters on. Set to None if not sharding.
        core_grid: Core grid to use for the group norm parameters. Must be provided if device is None
        return_mask: Whether to return the mask.
        dtype: Data type to use for the group norm parameters.
    Returns:
        The prepared group norm parameters in the same order as torch_params. If return_mask is True, returns masks.
        Examples: [weight, bias], mask if return_mask is True, [weight, bias] if return_mask is False for inputs [torch_weight, torch_bias]
            Input: [torch_weight, torch_bias]   Output: [tt_weight, tt_bias], tt_mask if return_mask is True, [tt_weight, tt_bias] if return_mask is False
            Input: torch_weight                 Output: tt_weight, tt_mask if return_mask is True, tt_weight if return_mask is False
    """
    import torch

    assert core_grid or device, "Either core_grid or device must be provided to determine virtual columns"
    assert (
        channels_per_device % 32 == 0 == channels_per_device % groups_per_device
    ), f"channels_per_device {channels_per_device} must be divisible by 32 and groups_per_device {groups_per_device}"

    num_devices = 1
    mapper_dims = [None, None]
    if mesh_axis is not None:
        num_devices = tuple(device.shape)[mesh_axis]
        mapper_dims[mesh_axis] = 0  # shadding on channel dimension

    # Calculate number of virtual columns that will be used
    dev_core_grid = core_grid or device.core_grid
    num_virtual_cols = dram_group_norm_virtual_columns(dev_core_grid, channels_per_device, groups_per_device)
    tt_params = []
    torch_params_itr = [torch_params] if isinstance(torch_params, torch.Tensor) else torch_params

    # Create prepared device tensors for group norm
    for torch_param in torch_params_itr:
        computed_channels_per_device = torch_param.numel() // num_devices
        assert (
            computed_channels_per_device == channels_per_device
        ), f"Computed number of channels per device: {computed_channels_per_device} not equal to provided number of channels per device: {channels_per_device}"
        torch_sharded_lst = [
            ttnn.create_group_norm_weight_bias_rm(t, channels_per_device, num_virtual_cols)
            for t in torch_param.chunk(num_devices)
        ]
        tensor_to_shard = torch.cat(torch_sharded_lst, dim=0)

        tt_params.append(
            ttnn.from_torch(
                tensor_to_shard,
                dtype=dtype,
                device=device,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=mapper_dims),
            )
        )

    tt_params = tt_params[0] if isinstance(torch_params, torch.Tensor) else tt_params
    if return_mask:
        tt_mask = ttnn.create_group_norm_input_mask(channels_per_device, groups_per_device, num_virtual_cols, dtype)
        tt_mask = ttnn.to_device(tt_mask, device)
        return tt_params, tt_mask
    else:
        return tt_params


def find_max_tile_span(W, group_size, tile_width):
    """Finds the maximum (worst case) number of tiles a group of size group_size can span across.
    This helps in setting the mask width conservatively.
    """
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


def create_group_norm_reciprocals_impl(N, C, H, W, num_groups, core_grid):
    """
    Create reciprocals tensor for group norm with welford algorithm.
    Generates reciprocal values 1/1, 1/2, 1/3, ..., 1/N.
    The number of elements is based on the tensor size and the number of groups.
    The tensor is replicated for each core so that when sharded to L1 memory, each core has a complete copy.

    Args:
        N: Batch size
        C: Number of channels
        H: Height
        W: Width
        num_groups: Number of groups
        core_grid: Core grid

    Returns:
        Row major tensor with reciprocal values
    """
    import torch

    num_virtual_cols = dram_group_norm_virtual_columns(core_grid, C, num_groups)
    num_virtual_rows = (core_grid.x // num_virtual_cols) * core_grid.y

    # Calculate batch distribution
    num_virtual_rows_per_group = 1 if N >= num_virtual_rows else num_virtual_rows // N
    num_channels_per_group = C // num_groups
    num_height_tiles_per_group = math.ceil(H * W / ttnn.TILE_SIZE)

    num_reciprocals_per_group = num_channels_per_group * num_height_tiles_per_group
    num_reciprocals_per_core = num_reciprocals_per_group // num_virtual_rows_per_group

    # Create reciprocal values: 1/1, 1/2, 1/3, ..., 1/max_n
    reciprocals_tensor = 1.0 / torch.arange(1, num_reciprocals_per_core + 1, dtype=torch.float32)

    # Repeat the reciprocals tensor for each core so they all have identical copies
    return reciprocals_tensor.repeat(core_grid.x * core_grid.y, 1)


def create_group_norm_reciprocals(N, C, H, W, num_groups, core_grid):
    return create_group_norm_reciprocals_impl(N, C, H, W, num_groups, core_grid)


def get_group_norm_cores_across_channel(memory_layout, core_grid, shard_orientation=None):
    """Compute effective cores that split the channel axis.

    For BLOCK_SHARDED, the channel axis lives in grid.y (COL_MAJOR)
    or grid.x (ROW_MAJOR).  When *shard_orientation* is not supplied
    the legacy COL_MAJOR behaviour is assumed.
    """
    if memory_layout == ttnn.types.TensorMemoryLayout.BLOCK_SHARDED:
        if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
            num_cores_across_channel = core_grid.x
        else:
            num_cores_across_channel = core_grid.y
    elif memory_layout == ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores_across_channel = 1
    else:
        num_cores_across_channel = core_grid.x * core_grid.y

    return num_cores_across_channel


def _golden_function(
    input_tensor: ttnn.Tensor,
    *,
    num_groups,
    epsilon=1e-05,
    weight=None,
    bias=None,
    memory_config=None,
    core_grid=None,
    input_mask=None,
    **kwargs,
):
    import torch

    num_channels = input_tensor.shape[-1]
    shard_orientation = getattr(memory_config.shard_spec, "orientation", None) if memory_config.shard_spec else None
    num_cores_across_channel = get_group_norm_cores_across_channel(
        memory_config.memory_layout, core_grid, shard_orientation
    )
    weight = weight.reshape((num_cores_across_channel, -1))
    weight = weight[:, : num_channels // num_cores_across_channel].flatten()
    if bias is not None:
        bias = bias.reshape((num_cores_across_channel, -1))
        bias = bias[:, : num_channels // num_cores_across_channel].flatten()

    input_tensor = input_tensor.permute(0, 3, 1, 2)
    output = torch.nn.functional.group_norm(input_tensor.float(), num_groups, weight.float(), bias.float(), eps=epsilon)
    output = output.permute(0, 2, 3, 1)
    return output


def _postprocess_golden_function_outputs(output, args, kwargs):
    input_tensor = args[0]
    output = ttnn.reshape(output, input_tensor.shape)
    return output


ttnn.attach_golden_function(
    ttnn.group_norm,
    golden_function=_golden_function,
    postprocess_golden_function_outputs=_postprocess_golden_function_outputs,
)

__all__ = []
