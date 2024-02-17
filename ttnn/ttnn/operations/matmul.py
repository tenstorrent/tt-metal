# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

from loguru import logger

import ttnn

MatmulDefaultProgramConfig = ttnn.ttl.operations.primary.MatmulDefaultProgramConfig
MatmulMultiCoreReuseProgramConfig = ttnn.ttl.operations.primary.MatmulMultiCoreReuseProgramConfig
MatmulMultiCoreReuseMultiCastProgramConfig = ttnn.ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig
MatmulMultiCoreReuseMultiCast1DProgramConfig = ttnn.ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig

# MatmulProgramConfig is the Union of the above types
MatmulProgramConfig = ttnn.ttl.operations.primary.MatmulProgramConfig


def map_num_cores_to_core_grid(num_cores: int, max_core_grid) -> ttnn.CoreGrid:
    for y in range(max_core_grid.y + 1, 1, -1):
        for x in range(max_core_grid.x + 1, 1, -1):
            if x * y == num_cores:
                return ttnn.CoreGrid(y, x)
    raise RuntimeError(
        f"Unable to map {num_cores} cores to a core grid with a maximum of {max_core_grid.num_cores} cores"
    )


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


def _get_subblock_sizes(m_tiles_per_core, n_tiles_per_core):
    for m_subblock_size, n_subblock_size in _DST_SUB_BLOCKS:
        if m_tiles_per_core % m_subblock_size == 0 and n_tiles_per_core % n_subblock_size == 0:
            return m_subblock_size, n_subblock_size
    raise RuntimeError(f"Unable to find subblock sizes for m_size={m_tiles_per_core} and n_size={n_tiles_per_core}")


_SUPPORTED_ACTIVATIONS = ["gelu", "relu"]


_ACTIVATION_TO_FUSED_ACTIVATION = {
    "gelu": (ttnn.ttl.tensor.FusibleActivation.GELU, True),
    "relu": ttnn.ttl.tensor.FusibleActivation.RELU,
}


def get_fused_activation(activation):
    if activation is None:
        return None
    return _ACTIVATION_TO_FUSED_ACTIVATION[activation]


def _validate_activation(activation):
    if activation is None:
        return
    is_supported = activation in _SUPPORTED_ACTIVATIONS
    if not is_supported:
        raise RuntimeError(
            f"{activation} is not supported as activation function. Use one of these instead: {_SUPPORTED_ACTIVATIONS}"
        )


def create_matmul_1d_systolic_array_config(
    *,
    input_shape_a: Tuple[int, ...],
    input_shape_b: Tuple[int, ...],
    max_core_grid: Optional[ttnn.CoreGrid] = None,
    activation: Optional[str] = None,
):
    """

    Create a MatmulMultiCoreReuseMultiCast1DProgramConfig for a 1D systolic array.


    Args:
        * :attr:`input_shape_a` (Tuple[int, ...]): the shape of the first tensor
        * :attr:`input_shape_b` (Tuple[int, ...]): the shape of the second tensor
        * :attr:`max_core_grid` (ttnn.CoreGrid): the maximum core grid to use
        * :attr:`activation` (Optional[str]): the activation function to use. Defaults to None

    """

    _validate_activation(activation)

    if max_core_grid is None:
        raise RuntimeError(f"core_grid must be a valid CoreGrid object")

    if max_core_grid is not None and not isinstance(max_core_grid, ttnn.CoreGrid):
        raise RuntimeError(f"core_grid must be a valid CoreGrid object")

    *batch_shape_a, m_size, k_size = input_shape_a
    *batch_shape_b, _, n_size = input_shape_b
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

    num_cores = max_core_grid.num_cores
    while k_tiles % num_cores != 0 or n_tiles % num_cores != 0:
        num_cores -= 1
    core_grid = map_num_cores_to_core_grid(num_cores, max_core_grid)
    if core_grid.num_cores == 1:
        raise RuntimeError(f"Cannot run this operation on a single core")

    batch_and_m_tiles_per_core = batch_and_m_tiles
    k_tiles_per_core = k_tiles // core_grid.num_cores
    n_tiles_per_core = n_tiles // core_grid.num_cores

    m_subblock_size, n_subblock_size = _get_subblock_sizes(batch_and_m_tiles, n_tiles_per_core)

    return MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=k_tiles_per_core,
        out_subblock_h=m_subblock_size,
        out_subblock_w=n_subblock_size,
        per_core_M=batch_and_m_tiles_per_core,
        per_core_N=n_tiles_per_core,
        fuse_batch=True,
        fused_activation=get_fused_activation(activation=activation),
        mcast_in0=True,
    )


def _get_matmul_program_config(
    *, operation_name, input_tensor_a, input_tensor_b, core_grid, activation, use_1d_systolic_array
):
    *batch_shape_a, m_size, k_size = input_tensor_a.shape
    *batch_shape_b, _, n_size = input_tensor_b.shape
    batch_size = math.prod(batch_shape_a)
    input_b_is_batched = math.prod(batch_shape_b) > 1

    input_tensor_a_memory_config = ttnn.get_memory_config(input_tensor_a)
    input_tensor_b_memory_config = ttnn.get_memory_config(input_tensor_b)

    if use_1d_systolic_array is not None:
        # TODO: infer if 1D systolic array can be used
        if use_1d_systolic_array:
            return create_matmul_1d_systolic_array_config(
                input_shape_a=input_tensor_a.shape,
                input_shape_b=input_tensor_b.shape,
                max_core_grid=core_grid,
                activation=activation,
            )

    # TODO: clean up the code below by mvoing it to separate create_*_config functions

    if m_size % ttnn.TILE_SIZE != 0 or k_size % ttnn.TILE_SIZE != 0:
        raise RuntimeError(
            f"{operation_name}: The last two dimensions of the first tensor must be a multiple of {ttnn.TILE_SIZE}"
        )

    if k_size % ttnn.TILE_SIZE != 0 or n_size % ttnn.TILE_SIZE != 0:
        raise RuntimeError(
            f"{operation_name}: The last two dimensions of the second tensor must be a multiple of {ttnn.TILE_SIZE}"
        )

    if input_b_is_batched:
        if (not ttnn.is_sharded(input_tensor_a)) and (not ttnn.is_sharded(input_tensor_b)):
            m_tiles_per_core = int(math.ceil((m_size / ttnn.TILE_SIZE)))
            n_tiles_per_core = int(math.ceil((n_size / ttnn.TILE_SIZE)))
            k_tiles_per_core = 1  # TODO(arakhmati): Can it be more than 1 without running out of memory?
        elif ttnn.is_sharded(input_tensor_a):
            if input_tensor_a_memory_config.memory_layout == ttnn.ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                raise RuntimeError(f"{operation_name}: Cannot be width sharded")
            shard_shape = input_tensor_a_memory_config.shard_spec.shape
            N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
            m_tiles_per_core = shard_shape[0] // ttnn.TILE_SIZE
            n_tiles_per_core = N
            k_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE
        elif ttnn.is_sharded(input_tensor_b):
            if input_tensor_b_memory_config.memory_layout == ttnn.ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                raise RuntimeError(f"{operation_name}: Cannot be width sharded")
            shard_shape = input_tensor_b_memory_config.shard_spec.shape
            m_tiles_per_core = int(math.ceil((m_size / ttnn.TILE_SIZE)))
            n_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE
            k_tiles_per_core = 1
    else:
        if not ttnn.is_sharded(input_tensor_a):
            m_tiles_per_core = int(math.ceil(((batch_size * m_size) / ttnn.TILE_SIZE) / core_grid.y))
            n_tiles_per_core = int(math.ceil(n_size / ttnn.TILE_SIZE / core_grid.x))
            k_tiles_per_core = 4  # TODO(arakhmati): What is a good starting point?
            while (k_size // ttnn.TILE_SIZE) % k_tiles_per_core != 0:
                k_tiles_per_core -= 1
        else:
            if not input_tensor_a_memory_config.memory_layout == ttnn.ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED:
                raise RuntimeError(f"{operation_name}: Must be block sharded")
            K = input_tensor_a.shape[-1] // ttnn.TILE_SIZE
            N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
            shard_shape = input_tensor_a_memory_config.shard_spec.shape
            m_tiles_per_core = shard_shape[0] // ttnn.TILE_SIZE
            n_tiles_per_core = (N * shard_shape[1]) // (K * ttnn.TILE_SIZE)
            k_tiles_per_core = 1

    m_subblock_size, n_subblock_size = _get_subblock_sizes(m_tiles_per_core, n_tiles_per_core)

    if input_b_is_batched:
        program_config = ttnn.ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            per_core_M=m_tiles_per_core,
            per_core_N=n_tiles_per_core,
            in0_block_w=k_tiles_per_core,
            out_subblock_h=m_subblock_size,
            out_subblock_w=n_subblock_size,
        )
    else:
        program_config = ttnn.ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
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


def _matmul(
    operation_name,
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[Tuple[int, int]] = None,
    program_config: Optional[MatmulProgramConfig] = None,
    activation: Optional[str] = None,
    use_1d_systolic_array: Optional[bool] = None,
) -> ttnn.Tensor:
    _validate_activation(activation)

    if core_grid is not None and program_config is not None:
        raise RuntimeError(f"{operation_name}: core_grid and program_config cannot be used together")

    if core_grid is not None and not isinstance(core_grid, ttnn.CoreGrid):
        raise RuntimeError(f"{operation_name}: core_grid must be a valid CoreGrid object")

    if use_1d_systolic_array is not None and core_grid is None:
        core_grid = input_tensor_a.device.core_grid

    if dtype is None:
        dtype = input_tensor_a.dtype

    input_shape_a = input_tensor_a.shape
    input_shape_b = input_tensor_b.shape

    output_shape_list = []
    padded_output_shape_list = []
    for index in range(len(input_shape_a) - 1):
        output_shape_list.append(input_shape_a[index])
        padded_output_shape_list.append(input_shape_a.with_tile_padding()[index])
    output_shape_list.append(input_shape_b[-1])
    padded_output_shape_list.append(input_shape_b.with_tile_padding()[-1])
    output_shape = ttnn.Shape(output_shape_list, padded_output_shape_list)

    *_, _, width_a = input_shape_a
    *batch_shape_b, height_b, _ = input_shape_b

    input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
    input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

    if bias is not None:
        bias = ttnn.unsqueeze_to_4D(bias)

    if width_a != height_b:
        raise RuntimeError(
            f"{operation_name}: The width of the first tensor must be equal to the height of the second tensor"
        )

    apply_activation_separately = False
    if core_grid is not None or program_config is not None:
        if core_grid is not None:
            if core_grid.num_cores == 1:
                raise RuntimeError(f"{operation_name}: core_grid must have more than 1 core")
            program_config = _get_matmul_program_config(
                operation_name=operation_name,
                input_tensor_a=input_tensor_a,
                input_tensor_b=input_tensor_b,
                core_grid=core_grid,
                activation=activation,
                use_1d_systolic_array=use_1d_systolic_array,
            )

        apply_activation_separately = not hasattr(program_config, "fused_activation")

        try:
            output_tensor = ttnn.ttl.operations.primary.matmul(
                input_tensor_a,
                input_tensor_b,
                bias=bias,
                program_config=program_config,
                output_mem_config=memory_config,
                output_dtype=dtype,
            )
        except Exception as exception:
            first_line_of_exception = "\n".join(str(exception).split("\n")[:3])
            raise RuntimeError(
                f"{operation_name}: ttl.operations.primary.matmul failed with error: {first_line_of_exception}"
            )

    else:
        apply_activation_separately = activation is not None
        if dtype != input_tensor_a.dtype:
            raise RuntimeError("dtype must be the same as the input tensors")

        input_b_is_batched = math.prod(batch_shape_b) > 1
        if input_b_is_batched:
            output_tensor = ttnn.ttl.tensor.bmm(
                input_tensor_a,
                input_tensor_b,
                output_mem_config=memory_config,
            )
        else:
            output_tensor = ttnn.ttl.tensor.matmul(
                input_tensor_a,
                input_tensor_b,
                output_mem_config=memory_config,
            )
            if bias is not None:
                output_tensor += bias

    if apply_activation_separately and activation is not None:
        activation_to_function = {
            "gelu": ttnn.gelu,
            "relu": ttnn.relu,
        }
        output_tensor = activation_to_function[activation](output_tensor)

    if output_tensor.shape != output_shape:
        output_tensor = ttnn.reshape(output_tensor, output_shape)
    return output_tensor


def _matmul_validate_input_tensors(operation_name, input_tensor_a, input_tensor_b, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor_a,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor_b,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def _torch_matmul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
    input_tensor_a = ttnn.from_device(input_tensor_a)
    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_a = ttnn.to_torch(input_tensor_a)

    input_tensor_b = ttnn.from_device(input_tensor_b)
    input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_b = ttnn.to_torch(input_tensor_b)

    return input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)


@ttnn.register_operation(
    name="ttnn.matmul", validate_input_tensors=_matmul_validate_input_tensors, torch_function=_torch_matmul
)
def matmul(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[Tuple[int, int]] = None,
    program_config: Optional[MatmulProgramConfig] = None,
    use_1d_systolic_array: Optional[bool] = None,
) -> ttnn.Tensor:
    """
    matmul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig=ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None, core_grid: Optional[Tuple[int, int]] = None, program_config: Optional[MatmulProgramConfig] = None, use_1d_systolic_array: Optional[bool] = None) -> ttnn.Tensor

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
        * :attr:`core_grid` (Tuple[int, int]): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to None
        * :attr:`program_config` (ttnn.MatmulProgramConfig): the program configuration for the matmul operation. Defaults to None
        * :attr:`use_1d_systolic_array` (bool): whether to use a 1D systolic array. Defaults to None which means it will be determined automatically

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

    return _matmul(
        "ttnn.matmul",
        input_tensor_a,
        input_tensor_b,
        memory_config=memory_config,
        dtype=dtype,
        core_grid=core_grid,
        program_config=program_config,
        use_1d_systolic_array=use_1d_systolic_array,
    )


def _linear_validate_input_tensors(operation_name, input_tensor_a, input_tensor_b, *args, bias=None, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor_a,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor_b,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        bias,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
        is_optional=True,
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


@ttnn.register_operation(
    name="ttnn.linear", validate_input_tensors=_linear_validate_input_tensors, torch_function=_torch_linear
)
def linear(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[Tuple[int, int]] = None,
    program_config: Optional[MatmulProgramConfig] = None,
    activation: Optional[str] = None,
    use_1d_systolic_array: Optional[bool] = None,
) -> ttnn.Tensor:
    """
    linear(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, bias: Optional[ttnn.Tensor] = None, memory_config: ttnn.MemoryConfig=ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None, core_grid: Optional[Tuple[int, int]] = None, proggram_config: Optional[MatmulProgramConfig] = None, activation: Optional[str] = None, use_1d_systolic_array: Optional[bool] = None) -> ttnn.Tensor

    Returns the linear transformation of the inputs

    Arguments:
        * :attr:`input_tensor_a` (ttnn.Tensor): the first tensor to be multiplied
        * :attr:`input_tensor_b` (ttnn.Tensor): the second tensor to be multiplied

    Keyword Arguments:
        * :attr:`bias` (Optional[ttnn.Tensor]): the bias tensor to be added. Defaults to None
        * :attr:`memory_config` (ttnn.MemoryConfig): the memory configuration of the output tensor. Defaults to ttnn.DRAM_MEMORY_CONFIG
        * :attr:`dtype` (Optional[ttnn.DataType]): the data type of the output tensor. Defaults to None
        * :attr:`core_grid` (Optional[Tuple[int, int]]): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to None
        * :attr:`program_config` (Optional[MatmulProgramConfig]): the program configuration for the matmul operation. Defaults to None
        * :attr:`activation` (Optional[str]): the activation function to be applied. Defaults to None
        * :attr:`use_1d_systolic_array` (Optional[bool]): whether to use 1D systolic array. Defaults to None which means it will be determined automatically

    Example::
        >>> # batched matrix x broadcasted matrix
        >>> activations = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> weight = ttnn.to_device(ttnn.from_torch(torch.randn((32, 128), dtype=torch.bfloat16)), device)
        >>> bias = ttnn.to_device(ttnn.from_torch(torch.randn((128,), dtype=torch.bfloat16)), device)
        >>> output = ttnn.linear(activations, weight, bias=bias)
        >>> print(output.shape)
        [10, 64, 128]
    """

    return _matmul(
        "ttnn.linear",
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        memory_config=memory_config,
        dtype=dtype,
        core_grid=core_grid,
        program_config=program_config,
        activation=activation,
        use_1d_systolic_array=use_1d_systolic_array,
    )


ttnn.Tensor.__matmul__ = matmul


__all__ = []
