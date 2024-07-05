# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import ttnn

MatmulProgramConfig = ttnn._tt_lib.operations.primary.MatmulProgramConfig
MatmulMultiCoreReuseProgramConfig = ttnn._tt_lib.operations.primary.MatmulMultiCoreReuseProgramConfig
MatmulMultiCoreReuseMultiCastProgramConfig = ttnn._tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig
MatmulMultiCoreReuseMultiCast1DProgramConfig = (
    ttnn._tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig
)
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig = (
    ttnn._tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
)


def _golden_function(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    if transpose_a:
        input_tensor_a = input_tensor_a.transpose(-1, -2)
    if transpose_b:
        input_tensor_b = input_tensor_b.transpose(-1, -2)
    output_tensor = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)

    if activation == "gelu":
        output_tensor = torch.nn.functional.gelu(output_tensor)
    elif activation == "relu":
        output_tensor = torch.nn.functional.relu(output_tensor)
    elif activation is not None:
        raise RuntimeError(f"{activation} is not supported as activation function")

    while len(output_tensor.shape) > len(input_tensor_a.shape):
        output_tensor = output_tensor.squeeze(0)
    return output_tensor


@ttnn.register_python_operation(name="ttnn.matmul", golden_function=_golden_function)
def matmul(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[ttnn.CoreGrid] = None,
    program_config: Optional[MatmulProgramConfig] = None,
    activation: Optional[str] = None,
    use_1d_systolic_array: Optional[bool] = None,
    compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
) -> ttnn.Tensor:
    """
    matmul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig=ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None, core_grid: Optional[ttnn.CoreGrid] = None, program_config: Optional[MatmulProgramConfig] = None, activation: Optional[str] = None, use_1d_systolic_array: Optional[bool] = None, compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None) -> ttnn.Tensor

    Returns the matrix product of two tensors.

    The behavior depends on the dimensionality of the tensors as follows:

    - If both arguments are 2-dimensional, the matrix-matrix product is returned.
    - If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension for the purpose of the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.
    - If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned in 2 dimensions.
    - If both arguments are at least 1-dimensional and at least one argument is
      N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
      argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
      batched matrix multiply.  If the second argument is 1-dimensional, a
      1 is appended to its dimension for the purpose of the batched matrix multiple.
      The non-matrix (i.e. batch) dimensions must be broadcastable.  For example, if :attr:`input_tensor_a` is a
      :math:`(j \\times 1 \\times n_size \\times n_size)` tensor and :attr:`input_tensor_b` is a :math:`(k_size \\times n_size \\times n_size)`
      tensor, the result will be a :math:`(j \\times k_size \\times n_size \\times n_size)` tensor.
    - In order to leverage sharded matmul implementations we can shard both input_tensor_a and input_tensor_b. The sharding strategy used will be according
      to the sharding stategy on the respective tensor. A sharded 1D matmul can be either HEIGHT or WIDTH sharded, 2D matmuls can be block sharded.

      Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
      are broadcastable, and not the matrix dimensions. For example, if :attr:`input_tensor_a` is a
      :math:`(j \\times 1 \\times n_size \\times m_size)` tensor and :attr:`input_tensor_b` is a :math:`(k_size \\times m_size \\times p)`
      tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
      matrix dimensions) are different. The operation will return a :math:`(j \\times k_size \\times n_size \\times p)` tensor.


    .. note::

        The 1-dimensional dot product version of this function is currently returning the Tensor with a non-empty shape. This is expected to be fixed in an upcomming release.

    Arguments:
        * :attr:`input_tensor_a` (ttnn.Tensorensor): the first tensor to be multiplied
        * :attr:`input_tensor_b` (ttnn.Tensor): the second tensor to be multiplied

    Keyword Arguments:
        * :attr:`memory_config` (ttnn.MemoryConfig): the memory configuration of the output tensor. Defaults to ttnn.DRAM_MEMORY_CONFIG
        * :attr:`dtype` (ttnn.DataType): the data type of the output tensor. Defaults to None
        * :attr:`core_grid` (ttnn.CoreGrid): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to None
        * :attr:`program_config` (ttnn.MatmulProgramConfig): the program configuration for the matmul operation. Defaults to None
        * :attr:`activation` (Optional[str]): the activation function to be applied. Defaults to None
        * :attr:`use_1d_systolic_array` (bool): whether to use a 1D systolic array. Defaults to None which means it will be determined automatically
        * :attr:`compute_kernel_config` (ttnn.DeviceComputeKernelConfig): the compute kernel configuration for the matmul operation. Defaults to None

    Example::

        >>> # vector x vector
        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((32), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32), dtype=torch.bfloat16)), device)
        >>> output = tensor1 @ tensor2
        >>> print(output.shape)
        [32]
        >>> # matrix x vector
        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((64, 32), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32), dtype=torch.bfloat16)), device)
        >>> output = tensor1 @ tensor2
        >>> print(output.shape)
        [64, 1]
        >>> # batched matrix x broadcasted vector
        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32), dtype=torch.bfloat16)), device)
        >>> output = tensor1 @ tensor2
        >>> print(output.shape)
        [10, 64, 1]
        >>> # batched matrix x batched matrix
        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 32, 128), dtype=torch.bfloat16)), device)
        >>> output = tensor1 @ tensor2
        >>> print(output.shape)
        [10, 64, 128]
        >>> # batched matrix x broadcasted matrix
        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32, 128), dtype=torch.bfloat16)), device)
        >>> output = tensor1 @ tensor2
        >>> print(output.shape)
        [10, 64, 128]
    """
    if use_1d_systolic_array is not None or core_grid is not None:
        if program_config is not None:
            raise RuntimeError(f"Cannot use program_config with use_1d_systolic_array or core_grid")
        core_grid = core_grid or input_tensor_a.device().core_grid

    return ttnn._ttnn.operations.matmul.matmul(
        input_tensor_a,
        input_tensor_b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
    )


def _golden_function(input_tensor_a, input_tensor_b, *, bias=None, activation=None, **kwargs):
    import torch

    if transpose_a:
        input_tensor_a = input_tensor_a.transpose(-1, -2)
    if transpose_b:
        input_tensor_b = input_tensor_b.transpose(-1, -2)
    output_tensor = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)

    if bias is not None:
        if len(bias) == 2:
            if bias.shape[0] != 1:
                raise RuntimeError(f"bias must be a 1D tensor")
            bias = bias[0]
        output_tensor += bias

    if activation == "gelu":
        output_tensor = torch.nn.functional.gelu(output_tensor)
    elif activation == "relu":
        output_tensor = torch.nn.functional.relu(output_tensor)
    elif activation is not None:
        raise RuntimeError(f"{activation} is not supported as activation function")

    while len(output_tensor.shape) > len(input_tensor_a.shape):
        output_tensor = output_tensor.squeeze(0)
    return output_tensor


@ttnn.register_python_operation(name="ttnn.linear", golden_function=_golden_function)
def linear(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: Optional[ttnn.Tensor] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[ttnn.CoreGrid] = None,
    program_config: Optional[MatmulProgramConfig] = None,
    activation: Optional[str] = None,
    use_1d_systolic_array: Optional[bool] = None,
    compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
) -> ttnn.Tensor:
    """
    linear(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, bias: Optional[ttnn.Tensor] = None, memory_config: ttnn.MemoryConfig=ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None, core_grid: Optional[ttnn.CoreGrid] = None, program_config: Optional[MatmulProgramConfig] = None, activation: Optional[str] = None, use_1d_systolic_array: Optional[bool] = None, compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None) -> ttnn.Tensor

    Returns the linear transformation of the inputs

    Arguments:
        * :attr:`input_tensor_a` (ttnn.Tensor): the first tensor to be multiplied
        * :attr:`input_tensor_b` (ttnn.Tensor): the second tensor to be multiplied

    Keyword Arguments:
        * :attr:`bias` (Optional[ttnn.Tensor]): the bias tensor to be added. Defaults to None
        * :attr:`memory_config` (ttnn.MemoryConfig): the memory configuration of the output tensor. Defaults to ttnn.DRAM_MEMORY_CONFIG
        * :attr:`dtype` (Optional[ttnn.DataType]): the data type of the output tensor. Defaults to None
        * :attr:`core_grid` (Optional[ttnn.CoreGrid]): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to None
        * :attr:`program_config` (Optional[MatmulProgramConfig]): the program configuration for the matmul operation. Defaults to None
        * :attr:`activation` (Optional[str]): the activation function to be applied. Defaults to None
        * :attr:`use_1d_systolic_array` (Optional[bool]): whether to use 1D systolic array. Defaults to None which means it will be determined automatically
        * :attr:`compute_kernel_config` (Optional[ttnn.DeviceComputeKernelConfig]): the compute kernel configuration for the matmul operation. Defaults to None

    Example::
        >>> # batched matrix x broadcasted matrix
        >>> activations = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> weight = ttnn.to_device(ttnn.from_torch(torch.randn((32, 128), dtype=torch.bfloat16)), device)
        >>> bias = ttnn.to_device(ttnn.from_torch(torch.randn((128,), dtype=torch.bfloat16)), device)
        >>> output = ttnn.linear(activations, weight, bias=bias)
        >>> print(output.shape)
        [10, 64, 128]
    """
    if use_1d_systolic_array is not None or core_grid is not None:
        if program_config is not None:
            raise RuntimeError(f"Cannot use program_config with use_1d_systolic_array or core_grid")
        core_grid = core_grid or input_tensor_a.device().core_grid

    # FIXME: passing an fp32 compute_kernel_config will cause the underlying C++ function to fail
    return ttnn._ttnn.operations.matmul.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
    )


ttnn.Tensor.__matmul__ = lambda self, *args, **kwargs: matmul(self, *args, **kwargs)


__all__ = []
