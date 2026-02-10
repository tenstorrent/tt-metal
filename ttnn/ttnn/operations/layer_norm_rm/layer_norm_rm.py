# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from ttnn.ttnn.operations.layer_norm_rm.layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
    *,
    device=None,
    memory_config=None,
) -> ttnn.Tensor:
    """
    Layer normalization with row-major layout.

    Computes layer normalization across the last dimension (W) for each row independently.
    The operation normalizes each row by subtracting the mean and dividing by the standard
    deviation, then applies learnable affine parameters gamma (scale) and beta (bias).

    Args:
        input_tensor: Input tensor in ROW_MAJOR layout, shape [..., H, W].
                     Must be on device with DRAM interleaved memory.
                     Dtype must be bfloat16 or float32.
                     H and W must be multiples of 32.
        gamma: Optional scale parameter, shape [1, ..., 1, W].
              Must match input dtype and be in ROW_MAJOR layout on device.
              If None, no scaling is applied (equivalent to gamma=1).
        beta: Optional bias parameter, shape [1, ..., 1, W].
             Must match input dtype and be in ROW_MAJOR layout on device.
             If None, no bias is applied (equivalent to beta=0).
        epsilon: Small constant for numerical stability in the denominator.
                Default is 1e-5.
        device: Device to execute on. Defaults to input_tensor's device.
        memory_config: Memory configuration for output tensor.
                      Defaults to DRAM_MEMORY_CONFIG.

    Returns:
        Output tensor with the same shape, dtype, and layout as input_tensor.

    Raises:
        RuntimeError: If input tensor requirements are not met (see spec).
    """

    # Validate input tensor
    if not input_tensor.is_allocated():
        raise RuntimeError("input must be on device")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError("input must be in ROW_MAJOR layout")

    if input_tensor.dtype not in [ttnn.bfloat16, ttnn.float32]:
        raise RuntimeError("unsupported dtype, must be bfloat16 or float32")

    # Get input shape
    input_shape = input_tensor.shape
    rank = len(input_shape)

    if rank < 2:
        raise RuntimeError("input must have rank >= 2")

    W = input_shape[-1]
    H = input_shape[-2]

    if W % 32 != 0:
        raise RuntimeError("last dimension must be a multiple of 32")

    if H % 32 != 0:
        raise RuntimeError("second-to-last dimension must be a multiple of 32")

    # Validate gamma if provided
    if gamma is not None:
        if not gamma.is_allocated():
            raise RuntimeError("gamma must be on device")

        if gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("gamma must be in ROW_MAJOR layout")

        if gamma.dtype != input_tensor.dtype:
            raise RuntimeError("gamma dtype must match input dtype")

        gamma_shape = gamma.shape
        if gamma_shape[-1] != W:
            raise RuntimeError("gamma last dim must match input last dim")

        # Check that all dims except last are 1
        for i in range(len(gamma_shape) - 1):
            if gamma_shape[i] != 1:
                raise RuntimeError("gamma must be broadcastable to input shape")

    # Validate beta if provided
    if beta is not None:
        if not beta.is_allocated():
            raise RuntimeError("beta must be on device")

        if beta.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("beta must be in ROW_MAJOR layout")

        if beta.dtype != input_tensor.dtype:
            raise RuntimeError("beta dtype must match input dtype")

        beta_shape = beta.shape
        if beta_shape[-1] != W:
            raise RuntimeError("beta last dim must match input last dim")

        # Check that all dims except last are 1
        for i in range(len(beta_shape) - 1):
            if beta_shape[i] != 1:
                raise RuntimeError("beta must be broadcastable to input shape")

    # Calculate output shape (same as input)
    output_shape = [input_shape[i] for i in range(len(input_shape))]

    # Select device
    target_device = device if device is not None else input_tensor.device()

    # Select memory config
    target_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Allocate output tensor on device
    # CRITICAL: Use positional args, not keyword args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape), input_tensor.dtype, input_tensor.layout, target_device, target_memory_config
    )

    # Create program descriptor
    program_descriptor = create_program_descriptor(input_tensor, gamma, beta, output_tensor, epsilon)

    # Execute - OUTPUT MUST BE LAST IN LIST
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
