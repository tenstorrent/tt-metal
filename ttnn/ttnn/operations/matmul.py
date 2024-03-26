# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import ttnn

MatmulDefaultProgramConfig = ttnn.experimental.operations.primary.MatmulDefaultProgramConfig
MatmulMultiCoreReuseProgramConfig = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig
MatmulMultiCoreReuseMultiCastProgramConfig = (
    ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig
)
MatmulMultiCoreReuseMultiCast1DProgramConfig = (
    ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig
)

# MatmulProgramConfig is the Union of the above types
MatmulProgramConfig = ttnn.experimental.operations.primary.MatmulProgramConfig


_DST_SUB_BLOCKS = [
    (2, 4),
    (4, 2),
    (1, 8),
    (8, 1),
    (1, 7),
    (7, 1),
    (2, 3),
    (3, 2),
    (1, 6),
    (6, 1),
    (1, 5),
    (5, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 3),
    (3, 1),
    (1, 2),
    (2, 1),
    (1, 1),
]

_FP32_DST_SUB_BLOCKS = [x for x in _DST_SUB_BLOCKS if x[0] * x[1] <= 4]


def _get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dst):
    candidate_sub_blocks = _FP32_DST_SUB_BLOCKS if fp32_dst else _DST_SUB_BLOCKS
    for m_subblock_size, n_subblock_size in candidate_sub_blocks:
        if m_tiles_per_core % m_subblock_size == 0 and n_tiles_per_core % n_subblock_size == 0:
            return m_subblock_size, n_subblock_size
    raise RuntimeError(
        f"Unable to find subblock sizes for m_size={m_tiles_per_core} and n_size={n_tiles_per_core} (fp32_dst={fp32_dst})"
    )


_ACTIVATION_TO_FUSED_ACTIVATION = {
    "gelu": (ttnn.experimental.tensor.FusibleActivation.GELU, True),
    "relu": ttnn.experimental.tensor.FusibleActivation.RELU,
    "silu": ttnn.experimental.tensor.FusibleActivation.SILU,
}


def get_fused_activation(activation):
    if activation is None:
        return None
    return _ACTIVATION_TO_FUSED_ACTIVATION[activation]


def _validate_activation(activation):
    if activation is None:
        return
    is_supported = activation in _ACTIVATION_TO_FUSED_ACTIVATION
    if not is_supported:
        raise RuntimeError(
            f"{activation} is not supported as activation function. Use one of these instead: {_ACTIVATION_TO_FUSED_ACTIVATION.keys()}"
        )


def create_matmul_1d_systolic_array_program_config(
    *,
    input_shape_a: Tuple[int, ...],
    input_shape_b: Tuple[int, ...],
    core_grid: Optional[ttnn.CoreGrid] = None,
    activation: Optional[str] = None,
    fp32_dst: Optional[bool] = False,
):
    """

    Create a MatmulMultiCoreReuseMultiCast1DProgramConfig for a 1D systolic array.


    Args:
        * :attr:`input_shape_a` (Tuple[int, ...]): the shape of the first tensor
        * :attr:`input_shape_b` (Tuple[int, ...]): the shape of the second tensor
        * :attr:`core_grid` (ttnn.CoreGrid): the maximum core grid to use
        * :attr:`activation` (Optional[str]): the activation function to use. Defaults to None

    """

    _validate_activation(activation)

    if core_grid is None:
        raise RuntimeError(f"core_grid must be a valid CoreGrid object")

    if core_grid is not None and not isinstance(core_grid, ttnn.CoreGrid):
        raise RuntimeError(f"core_grid must be a valid CoreGrid object")

    *batch_shape_a, m_size, k_size = input_shape_a.with_tile_padding()
    *batch_shape_b, _, n_size = input_shape_b.with_tile_padding()
    if math.prod(batch_shape_b) != 1:
        raise RuntimeError("Second input cannot be currently batched when running matmul using 1d systolic array")

    batch_size = math.prod(batch_shape_a)
    input_b_is_batched = math.prod(batch_shape_b) > 1

    if input_b_is_batched:
        raise RuntimeError(f"Input b shouldn't be batched")

    if (batch_size * m_size) % ttnn.TILE_SIZE != 0 or k_size % ttnn.TILE_SIZE != 0 or n_size % ttnn.TILE_SIZE != 0:
        raise RuntimeError(
            f"The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of {ttnn.TILE_SIZE}"
        )

    batch_and_m_tiles = (batch_size * m_size) // ttnn.TILE_SIZE
    k_tiles = k_size // ttnn.TILE_SIZE
    n_tiles = n_size // ttnn.TILE_SIZE

    is_tall = batch_and_m_tiles > n_tiles
    is_wide = not is_tall
    # Tall output
    if is_tall:
        batch_and_m_tiles_per_core = int(math.ceil(batch_and_m_tiles / core_grid.num_cores))
        k_tiles_per_core = int(math.ceil(k_tiles / core_grid.num_cores))
        n_tiles_per_core = n_tiles

    # Wide output
    else:
        batch_and_m_tiles_per_core = batch_and_m_tiles
        k_tiles_per_core = int(math.ceil(k_tiles / core_grid.num_cores))
        n_tiles_per_core = int(math.ceil(n_tiles / core_grid.num_cores))

    while k_tiles % k_tiles_per_core != 0:
        k_tiles_per_core -= 1

    m_subblock_size, n_subblock_size = _get_subblock_sizes(batch_and_m_tiles_per_core, n_tiles_per_core, fp32_dst)

    return MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=k_tiles_per_core,
        out_subblock_h=m_subblock_size,
        out_subblock_w=n_subblock_size,
        per_core_M=batch_and_m_tiles_per_core,
        per_core_N=n_tiles_per_core,
        fuse_batch=True,
        fused_activation=get_fused_activation(activation=activation),
        mcast_in0=is_wide,
    )


def create_matmul_program_config(
    *, input_tensor_a, input_tensor_b, core_grid, activation, use_1d_systolic_array, compute_kernel_config
):
    *batch_shape_a, m_size, k_size = input_tensor_a.shape.with_tile_padding()
    *batch_shape_b, _, n_size = input_tensor_b.shape.with_tile_padding()
    *_, intended_k_size_of_a = input_tensor_a.shape
    *_, intended_k_size_of_b, _ = input_tensor_b.shape

    if intended_k_size_of_a != intended_k_size_of_b:
        raise RuntimeError(f"The k dimension does not match between tensors")

    batch_size = math.prod(batch_shape_a)
    input_b_is_batched = math.prod(batch_shape_b) > 1

    input_tensor_a_memory_config = ttnn.get_memory_config(input_tensor_a)
    input_tensor_b_memory_config = ttnn.get_memory_config(input_tensor_b)

    # Determine how many subblock tiles we can use based on dest register data format
    fp32_dst = (
        compute_kernel_config
        and isinstance(compute_kernel_config, ttnn.WormholeComputeKernelConfig)
        and compute_kernel_config.fp32_dest_acc_en
    )

    if use_1d_systolic_array is None and not input_b_is_batched:
        # Infer use_1d_systolic_array based on how rectangular the output matrix
        height_width_ratio = (math.prod(batch_shape_a) * m_size) / n_size
        if height_width_ratio < 1:
            height_width_ratio = 1 / height_width_ratio

        # 8 is an arbitrary choice. It should probably be inferred based on the device core grid
        threshold_of_being_rectangular = 8
        is_more_rectangular_than_square = height_width_ratio > threshold_of_being_rectangular
        use_1d_systolic_array = is_more_rectangular_than_square

    if use_1d_systolic_array:
        return create_matmul_1d_systolic_array_program_config(
            input_shape_a=input_tensor_a.shape,
            input_shape_b=input_tensor_b.shape,
            core_grid=core_grid,
            activation=activation,
            fp32_dst=fp32_dst,
        )

    # TODO: clean up the code below by mvoing it to separate create_*_config functions

    if (batch_size * m_size) % ttnn.TILE_SIZE != 0 or k_size % ttnn.TILE_SIZE != 0 or n_size % ttnn.TILE_SIZE != 0:
        raise RuntimeError(
            f"The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of {ttnn.TILE_SIZE}"
        )

    if input_b_is_batched:
        if activation is not None:
            raise RuntimeError(f"Cannot use activation with batched input b")
        if (not ttnn.is_sharded(input_tensor_a)) and (not ttnn.is_sharded(input_tensor_b)):
            m_tiles_per_core = int(math.ceil((m_size / ttnn.TILE_SIZE)))
            n_tiles_per_core = int(math.ceil((n_size / ttnn.TILE_SIZE)))
            k_tiles_per_core = 1  # TODO(arakhmati): Can it be more than 1 without running out of memory?
        elif ttnn.is_sharded(input_tensor_a):
            if input_tensor_a_memory_config.memory_layout == ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                raise RuntimeError(f"MatmulMultiCoreReuseProgramConfig: Cannot be width sharded")
            shard_shape = input_tensor_a_memory_config.shard_spec.shape
            N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
            m_tiles_per_core = shard_shape[0] // ttnn.TILE_SIZE
            n_tiles_per_core = N
            k_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE
        elif ttnn.is_sharded(input_tensor_b):
            if input_tensor_b_memory_config.memory_layout == ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                raise RuntimeError(f"MatmulMultiCoreReuseProgramConfig: Cannot be width sharded")
            shard_shape = input_tensor_b_memory_config.shard_spec.shape
            m_tiles_per_core = int(math.ceil((m_size / ttnn.TILE_SIZE)))
            n_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE
            k_tiles_per_core = 1

        m_subblock_size, n_subblock_size = _get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dst)

        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            per_core_M=m_tiles_per_core,
            per_core_N=n_tiles_per_core,
            in0_block_w=k_tiles_per_core,
            out_subblock_h=m_subblock_size,
            out_subblock_w=n_subblock_size,
        )
    else:
        if not ttnn.is_sharded(input_tensor_a):
            m_tiles_per_core = int(math.ceil(((batch_size * m_size) / ttnn.TILE_SIZE) / core_grid.y))
            n_tiles_per_core = int(math.ceil(n_size / ttnn.TILE_SIZE / core_grid.x))
            k_tiles_per_core = 4  # TODO(arakhmati): What is a good starting point?
            while (k_size // ttnn.TILE_SIZE) % k_tiles_per_core != 0:
                k_tiles_per_core -= 1
        else:
            if (
                not input_tensor_a_memory_config.memory_layout
                == ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED
            ):
                raise RuntimeError(f"MatmulMultiCoreReuseMultiCastProgramConfig: Must be block sharded")
            K = input_tensor_a.shape[-1] // ttnn.TILE_SIZE
            N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
            shard_shape = input_tensor_a_memory_config.shard_spec.shape
            m_tiles_per_core = shard_shape[0] // ttnn.TILE_SIZE
            n_tiles_per_core = (N * shard_shape[1]) // (K * ttnn.TILE_SIZE)
            k_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE

        m_subblock_size, n_subblock_size = _get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dst)

        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            per_core_M=m_tiles_per_core,
            per_core_N=n_tiles_per_core,
            in0_block_w=k_tiles_per_core,
            out_subblock_h=m_subblock_size,
            out_subblock_w=n_subblock_size,
            transpose_mcast=False,
            fused_activation=get_fused_activation(activation=activation),
        )

    return program_config


def _torch_matmul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
    input_tensor_a = ttnn.from_device(input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_a = ttnn.to_torch(input_tensor_a)

    input_tensor_b = ttnn.from_device(input_tensor_b)
    input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_b = ttnn.to_torch(input_tensor_b)

    return input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)


@ttnn.register_operation(name="ttnn.matmul", is_cpp_function=True, torch_function=_torch_matmul)
def matmul(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[ttnn.CoreGrid] = None,
    program_config: Optional[MatmulProgramConfig] = None,
    use_1d_systolic_array: Optional[bool] = None,
    compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
) -> ttnn.Tensor:
    """
    matmul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig=ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None, core_grid: Optional[ttnn.CoreGrid] = None, program_config: Optional[MatmulProgramConfig] = None, use_1d_systolic_array: Optional[bool] = None, compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None) -> ttnn.Tensor

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
        program_config = create_matmul_program_config(
            input_tensor_a=input_tensor_a,
            input_tensor_b=input_tensor_b,
            core_grid=core_grid or input_tensor_a.device().core_grid,
            activation=None,
            use_1d_systolic_array=use_1d_systolic_array,
            compute_kernel_config=compute_kernel_config,
        )

    if program_config is not None:
        return ttnn._ttnn.operations.matmul.matmul(
            input_tensor_a,
            input_tensor_b,
            memory_config=memory_config,
            dtype=dtype,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    return ttnn._ttnn.operations.matmul.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=memory_config,
        dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )


def _torch_linear(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, bias=None, activation=None, **_):
    import torch

    input_tensor_a = ttnn.from_device(input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_a = ttnn.to_torch(input_tensor_a)

    input_tensor_b = ttnn.from_device(input_tensor_b)
    input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_b = ttnn.to_torch(input_tensor_b)

    output_tensor = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)

    if bias is not None:
        bias = ttnn.from_device(bias)
        bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.to_torch(bias, torch_rank=1)
        output_tensor += bias

    if activation == "gelu":
        output_tensor = torch.nn.functional.gelu(output_tensor)
    elif activation == "relu":
        output_tensor = torch.nn.functional.relu(output_tensor)
    elif activation is not None:
        raise RuntimeError(f"{activation} is not supported as activation function")

    return output_tensor


@ttnn.register_operation(name="ttnn.linear", is_cpp_function=True, torch_function=_torch_linear)
def linear(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[ttnn.CoreGrid] = None,
    program_config: Optional[MatmulProgramConfig] = None,
    activation: Optional[str] = None,
    use_1d_systolic_array: Optional[bool] = None,
    compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
) -> ttnn.Tensor:
    """
    linear(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, bias: Optional[ttnn.Tensor] = None, memory_config: ttnn.MemoryConfig=ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None, core_grid: Optional[ttnn.CoreGrid] = None, proggram_config: Optional[MatmulProgramConfig] = None, activation: Optional[str] = None, use_1d_systolic_array: Optional[bool] = None, compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None) -> ttnn.Tensor

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
        program_config = create_matmul_program_config(
            input_tensor_a=input_tensor_a,
            input_tensor_b=input_tensor_b,
            core_grid=core_grid or input_tensor_a.device().core_grid,
            activation=activation,
            use_1d_systolic_array=use_1d_systolic_array,
            compute_kernel_config=compute_kernel_config,
        )

    if program_config is not None:
        return ttnn._ttnn.operations.matmul.linear(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            memory_config=memory_config,
            dtype=dtype,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    # FIXME: passing an fp32 compute_kernel_config will cause the underlying C++ function to fail
    return ttnn._ttnn.operations.matmul.linear(
        input_tensor_a,
        input_tensor_b,
        memory_config=memory_config,
        bias=bias,
        dtype=dtype,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
    )


ttnn.Tensor.__matmul__ = lambda self, *args, **kwargs: matmul(self, *args, **kwargs)


__all__ = []
