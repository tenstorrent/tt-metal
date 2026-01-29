# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.experimental.programs module for creating program descriptors.

This module provides functions to create ProgramBranch objects for various operations.
These can be composed and executed together using ttnn.experimental.launch_composite().
"""

from typing import List, NamedTuple, Optional

import ttnn


class ProgramBranch(NamedTuple):
    """
    A single operation branch for parallel composition.

    Contains:
    - descriptor: The ProgramDescriptor for the operation
    - output: The output tensor (created by the operation's create_output_tensors)
    - io_tensors: All tensors (inputs + output) for generic_op

    Created by programs.rms_norm(), programs.layer_norm(), etc.
    Passed to launch_composite() for execution.

    Example:
        >>> left = ttnn.experimental.programs.rms_norm(input1, weight=w1)
        >>> right = ttnn.experimental.programs.rms_norm(input2, weight=w2)
        >>> left_out, right_out = ttnn.experimental.launch_composite([left, right])
    """

    descriptor: "ttnn.ProgramDescriptor"
    output: "ttnn.Tensor"
    io_tensors: List["ttnn.Tensor"]


def _create_program_config(input_tensor: "ttnn.Tensor") -> "ttnn.LayerNormProgramConfig":
    """
    Create appropriate program config based on input tensor's shard spec.

    For sharded inputs, creates LayerNormShardedMultiCoreProgramConfig.
    For non-sharded inputs, creates LayerNormDefaultProgramConfig.
    """
    if not input_tensor.is_sharded():
        return ttnn.LayerNormDefaultProgramConfig(use_welford=False)

    # Create sharded config from shard spec
    shard_spec = input_tensor.memory_config().shard_spec
    shard_shape = shard_spec.shape
    block_h = shard_shape[0] // 32
    block_w = shard_shape[1] // 32
    bbox = shard_spec.grid.bounding_box()
    grid_size = ttnn.CoreCoord(bbox.end.x - bbox.start.x + 1, bbox.end.y - bbox.start.y + 1)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=min(block_w, 4),
        block_h=block_h,
        block_w=block_w,
        inplace=True,
        use_welford=False,
    )


def _build_io_tensors(
    input_tensor: "ttnn.Tensor",
    output_tensor: "ttnn.Tensor",
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
) -> List["ttnn.Tensor"]:
    """Build the io_tensors list for a layernorm/rmsnorm operation."""
    io_tensors = [input_tensor]
    if residual_input_tensor is not None:
        io_tensors.append(residual_input_tensor)
    if weight is not None:
        io_tensors.append(weight)
    if bias is not None:
        io_tensors.append(bias)
    io_tensors.append(output_tensor)
    return io_tensors


def rms_norm(
    input_tensor: "ttnn.Tensor",
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    epsilon: float = 1e-12,
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
) -> ProgramBranch:
    """
    Create a ProgramBranch for an RMS norm operation.

    Automatically selects the appropriate factory (sharded or non-sharded) based on
    the input tensor's memory layout. Output tensor is created internally.

    Args:
        input_tensor: The input tensor (must be on device).
        core_range_set: The set of cores to run the operation on.
            For non-sharded: required - specifies which cores to use.
            For sharded: optional - if provided, validates that shard spec cores are within this range.
        epsilon: Small constant for numerical stability (default: 1e-12).
        weight: Optional weight (gamma) tensor for scaling.
        bias: Optional bias (beta) tensor for shifting.
        residual_input_tensor: Optional residual tensor to add before normalization.
        compute_kernel_config: Optional compute kernel configuration.
        memory_config: Optional output memory configuration. Defaults to input's memory config.

    Returns:
        ProgramBranch containing the program descriptor, output tensor, and io_tensors.
        Use with ttnn.experimental.launch_composite() to execute.

    Example:
        >>> left = ttnn.experimental.programs.rms_norm(input1, weight=w1)
        >>> right = ttnn.experimental.programs.rms_norm(input2, weight=w2)
        >>> left_out, right_out = ttnn.experimental.launch_composite([left, right])
    """
    device = input_tensor.device()
    arch = device.arch()

    # For non-sharded inputs, core_range_set is required
    if not input_tensor.is_sharded() and core_range_set is None:
        raise RuntimeError("core_range_set is required for non-sharded input tensors")

    # Initialize compute kernel config if not provided
    if compute_kernel_config is None:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    # Use input's memory config if not provided
    output_mem_config = memory_config if memory_config is not None else input_tensor.memory_config()

    # Create appropriate program config based on input tensor
    program_config = _create_program_config(input_tensor)

    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = ttnn.LayerNormType.RMSNORM
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = epsilon
    operation_params.output_mem_config = output_mem_config
    operation_params.program_config = program_config
    operation_params.compute_kernel_config = compute_kernel_config

    # Create LayerNormInputs
    tensor_args = ttnn.LayerNormInputs()
    tensor_args.input = input_tensor
    if residual_input_tensor is not None:
        tensor_args.residual_input_tensor = residual_input_tensor
    if weight is not None:
        tensor_args.weight = weight
    if bias is not None:
        tensor_args.bias = bias

    # Create output tensor using the device operation's create_output_tensors
    output_tensor = ttnn.LayerNormDeviceOperation.create_output_tensors(operation_params, tensor_args)

    # Select the appropriate factory and create descriptor
    factory = ttnn.LayerNormDeviceOperation.select_program_factory(operation_params, tensor_args)
    program_descriptor = factory.create_descriptor(operation_params, tensor_args, output_tensor, core_range_set)

    # Build io_tensors list
    io_tensors = _build_io_tensors(input_tensor, output_tensor, weight, bias, residual_input_tensor)

    return ProgramBranch(program_descriptor, output_tensor, io_tensors)


def layer_norm(
    input_tensor: "ttnn.Tensor",
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    epsilon: float = 1e-12,
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
) -> ProgramBranch:
    """
    Create a ProgramBranch for a layer norm operation.

    Automatically selects the appropriate factory (sharded or non-sharded) based on
    the input tensor's memory layout. Output tensor is created internally.

    Args:
        input_tensor: The input tensor (must be on device).
        core_range_set: The set of cores to run the operation on.
            For non-sharded: required - specifies which cores to use.
            For sharded: optional - if provided, validates that shard spec cores are within this range.
        epsilon: Small constant for numerical stability (default: 1e-12).
        weight: Optional weight (gamma) tensor for scaling.
        bias: Optional bias (beta) tensor for shifting.
        residual_input_tensor: Optional residual tensor to add before normalization.
        compute_kernel_config: Optional compute kernel configuration.
        memory_config: Optional output memory configuration. Defaults to input's memory config.

    Returns:
        ProgramBranch containing the program descriptor, output tensor, and io_tensors.
        Use with ttnn.experimental.launch_composite() to execute.

    Example:
        >>> left = ttnn.experimental.programs.layer_norm(input1, weight=w1, bias=b1)
        >>> right = ttnn.experimental.programs.layer_norm(input2, weight=w2, bias=b2)
        >>> left_out, right_out = ttnn.experimental.launch_composite([left, right])
    """
    device = input_tensor.device()
    arch = device.arch()

    # For non-sharded inputs, core_range_set is required
    if not input_tensor.is_sharded() and core_range_set is None:
        raise RuntimeError("core_range_set is required for non-sharded input tensors")

    # Initialize compute kernel config if not provided
    if compute_kernel_config is None:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    # Use input's memory config if not provided
    output_mem_config = memory_config if memory_config is not None else input_tensor.memory_config()

    # Create appropriate program config based on input tensor
    program_config = _create_program_config(input_tensor)

    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = ttnn.LayerNormType.LAYERNORM
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = epsilon
    operation_params.output_mem_config = output_mem_config
    operation_params.program_config = program_config
    operation_params.compute_kernel_config = compute_kernel_config

    # Create LayerNormInputs
    tensor_args = ttnn.LayerNormInputs()
    tensor_args.input = input_tensor
    if residual_input_tensor is not None:
        tensor_args.residual_input_tensor = residual_input_tensor
    if weight is not None:
        tensor_args.weight = weight
    if bias is not None:
        tensor_args.bias = bias

    # Create output tensor using the device operation's create_output_tensors
    output_tensor = ttnn.LayerNormDeviceOperation.create_output_tensors(operation_params, tensor_args)

    # Select the appropriate factory and create descriptor
    factory = ttnn.LayerNormDeviceOperation.select_program_factory(operation_params, tensor_args)
    program_descriptor = factory.create_descriptor(operation_params, tensor_args, output_tensor, core_range_set)

    # Build io_tensors list
    io_tensors = _build_io_tensors(input_tensor, output_tensor, weight, bias, residual_input_tensor)

    return ProgramBranch(program_descriptor, output_tensor, io_tensors)


__all__ = ["ProgramBranch", "rms_norm", "layer_norm"]
