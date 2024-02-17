# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

from loguru import logger

import ttnn

DST_SUB_BLOCKS = [
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


def _matmul(
    operation_name,
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: Optional[ttnn.DataType] = None,
    core_grid: Optional[Tuple[int, int]] = None,
    activation: Optional[str] = None,
) -> ttnn.Tensor:
    if core_grid is not None and not isinstance(core_grid, ttnn.CoreGrid):
        raise RuntimeError(f"{operation_name}: core_grid must be a valid CoreGrid object")

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

    # The idea is to make the shapes "possibly" broadcastable.
    if len(input_tensor_a.shape) > 4:
        raise RuntimeError(f"{operation_name}: There is currently no support for ranks greater than 4.")

    if len(input_shape_b) > 4:
        raise RuntimeError(f"{operation_name}: There is currently no support for ranks greater than {4}.")

    if len(input_shape_a) == 1:
        batch_shape_a = []
        height_a = 1
        (width_a,) = input_shape_a
    else:
        *batch_shape_a, height_a, width_a = input_shape_a

    if len(input_shape_b) == 1:
        batch_shape_b = []
        (height_b,) = input_shape_b
        width_b = 1
    else:
        *batch_shape_b, height_b, width_b = input_shape_b

    input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
    input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

    if bias is not None:
        bias = ttnn.unsqueeze_to_4D(bias)

    if width_a != height_b:
        raise RuntimeError(
            f"{operation_name}: The width of the first tensor must be equal to the height of the second tensor"
        )

    m_size = height_a
    k_size = width_a
    n_size = width_b

    batch_size = math.prod(batch_shape_a)
    input_b_has_batch = math.prod(batch_shape_b) > 1

    input_tensor_a_memory_config = ttnn.get_memory_config(input_tensor_a)
    input_tensor_b_memory_config = ttnn.get_memory_config(input_tensor_b)

    if core_grid is not None:
        if m_size % ttnn.TILE_SIZE != 0 or k_size % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                f"{operation_name}: The last two dimensions of the first tensor must be a multiple of 32"
            )

        if k_size % ttnn.TILE_SIZE != 0 or n_size % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                f"{operation_name}: The last two dimensions of the second tensor must be a multiple of 32"
            )

        if input_b_has_batch:
            if (not ttnn.is_sharded(input_tensor_a)) and (not ttnn.is_sharded(input_tensor_b)):
                y_tiles_per_core = int(math.ceil((m_size / ttnn.TILE_SIZE)))
                x_tiles_per_core = int(math.ceil((n_size / ttnn.TILE_SIZE)))
                k_block_size = 1  # TODO(arakhmati): Can it be more than 1 without running out of memory?
            elif ttnn.is_sharded(input_tensor_a):
                if input_tensor_a_memory_config.memory_layout == ttnn.ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                    raise RuntimeError(f"{operation_name}: Cannot be width sharded")
                shard_shape = input_tensor_a_memory_config.shard_spec.shape
                N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
                y_tiles_per_core = shard_shape[0] // ttnn.TILE_SIZE
                x_tiles_per_core = N
                k_block_size = shard_shape[1] // ttnn.TILE_SIZE
            elif ttnn.is_sharded(input_tensor_b):
                if input_tensor_b_memory_config.memory_layout == ttnn.ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                    raise RuntimeError(f"{operation_name}: Cannot be width sharded")
                shard_shape = input_tensor_b_memory_config.shard_spec.shape
                y_tiles_per_core = int(math.ceil((m_size / ttnn.TILE_SIZE)))
                x_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE
                k_block_size = 1
        else:
            if not ttnn.is_sharded(input_tensor_a):
                y_tiles_per_core = int(math.ceil(((batch_size * m_size) / ttnn.TILE_SIZE) / core_grid.y))
                x_tiles_per_core = int(math.ceil(n_size / ttnn.TILE_SIZE / core_grid.x))
                k_block_size = 4  # TODO(arakhmati): What is a good starting point?
                while (k_size // ttnn.TILE_SIZE) % k_block_size != 0:
                    k_block_size -= 1
            else:
                if not input_tensor_a_memory_config.memory_layout == ttnn.ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED:
                    raise RuntimeError(f"{operation_name}: Must be block sharded")
                K = input_tensor_a.shape[-1] // ttnn.TILE_SIZE
                N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
                shard_shape = input_tensor_a_memory_config.shard_spec.shape
                y_tiles_per_core = shard_shape[0] // ttnn.TILE_SIZE
                x_tiles_per_core = (N * shard_shape[1]) // (K * ttnn.TILE_SIZE)
                k_block_size = 1

        for m_subblock_size, n_subblock_size in DST_SUB_BLOCKS:
            if y_tiles_per_core % m_subblock_size == 0 and x_tiles_per_core % n_subblock_size == 0:
                break

        # logger.debug(f"input_b_has_batch={input_b_has_batch}")
        # logger.debug(f"y_tiles_per_core={y_tiles_per_core}")
        # logger.debug(f"x_tiles_per_core={x_tiles_per_core}")
        # logger.debug(f"k_block_size={k_block_size}")
        # logger.debug(f"m_subblock_size={m_subblock_size}")
        # logger.debug(f"n_subblock_size={n_subblock_size}")

        run_bias_separately = False
        run_activation_separately = False

        if input_b_has_batch:
            run_bias_separately = bias is not None
            run_activation_separately = activation is not None

            program_config = ttnn.ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(core_grid.x, core_grid.y),
                per_core_M=y_tiles_per_core,
                per_core_N=x_tiles_per_core,
                in0_block_w=k_block_size,  # k
                out_subblock_h=m_subblock_size,  # m
                out_subblock_w=n_subblock_size,  # n
            )
        else:
            if activation == "gelu":
                fused_activation = (ttnn.ttl.tensor.FusibleActivation.GELU, True)
            elif activation == "relu":
                fused_activation = ttnn.ttl.tensor.FusibleActivation.RELU
            elif activation is None:
                fused_activation = None
            else:
                raise RuntimeError(f"{activation} is not supported as activation function")

            program_config = ttnn.ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(core_grid.x, core_grid.y),
                per_core_M=y_tiles_per_core,
                per_core_N=x_tiles_per_core,
                in0_block_w=k_block_size,  # k
                out_subblock_h=m_subblock_size,  # m
                out_subblock_w=n_subblock_size,  # n
                transpose_mcast=False,
                fused_activation=fused_activation,
            )

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
        run_bias_separately = bias is not None
        run_activation_separately = activation is not None
        if dtype != input_tensor_a.dtype:
            raise RuntimeError("dtype must be the same as the input tensors")
        if input_b_has_batch:
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

    if run_bias_separately and bias is not None:
        output_tensor += bias

    if run_activation_separately and activation is not None:
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
) -> ttnn.Tensor:
    """
    matmul(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig=ttnn.DRAM_MEMORY_CONFIG, core_grid: Optional[Tuple[int, int]] = None) -> ttnn.Tensor

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
      :math:`(j \\times 1 \\times n \\times n)` tensor and :attr:`input_tensor_b` is a :math:`(k \\times n \\times n)`
      tensor, the result will be a :math:`(j \\times k \\times n \\times n)` tensor.
    - In order to leverage sharded matmul implementations we can shard both input_tensor_a and input_tensor_b. The sharding strategy used will be according
      to the sharding stategy on the respective tensor. A sharded 1D matmul can be either HEIGHT or WIDTH sharded, 2D matmuls can be block sharded.

      Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
      are broadcastable, and not the matrix dimensions. For example, if :attr:`input_tensor_a` is a
      :math:`(j \\times 1 \\times n \\times m)` tensor and :attr:`input_tensor_b` is a :math:`(k \\times m \\times p)`
      tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
      matrix dimensions) are different. The operation will return a :math:`(j \\times k \\times n \\times p)` tensor.


    .. note::

        The 1-dimensional dot product version of this function is currently returning the Tensor with a non-empty shape. This is expected to be fixed in an upcomming release.

    Arguments:
        * :attr:`input_tensor_a` (ttnn.Tensorensor): the first tensor to be multiplied
        * :attr:`input_tensor_b` (ttnn.Tensor): the second tensor to be multiplied

    Keyword Arguments:
        * :attr:`memory_config` (ttnn.MemoryConfig): the memory configuration of the output tensor. Defaults to ttnn.DRAM_MEMORY_CONFIG
        * :attr:`dtype` (ttnn.DataType): the data type of the output tensor. Defaults to None
        * :attr:`core_grid` (Tuple[int, int]): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to None

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
        "ttnn.matmul", input_tensor_a, input_tensor_b, memory_config=memory_config, dtype=dtype, core_grid=core_grid
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
    activation: Optional[str] = None,
) -> ttnn.Tensor:
    """
    linear(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, bias: Optional[ttnn.Tensor] = None, memory_config: ttnn.MemoryConfig=ttnn.DRAM_MEMORY_CONFIG, dtype: Optional[ttnn.DataType] = None, core_grid: Optional[Tuple[int, int]] = None, activation: Optional[str] = None) -> ttnn.Tensor

    Returns the linear transformation of the inputs

    Arguments:
        * :attr:`input_tensor_a` (ttnn.Tensor): the first tensor to be multiplied
        * :attr:`input_tensor_b` (ttnn.Tensor): the second tensor to be multiplied

    Keyword Arguments:
        * :attr:`bias` (Optional[ttnn.Tensor]): the bias tensor to be added. Defaults to None
        * :attr:`memory_config` (ttnn.MemoryConfig): the memory configuration of the output tensor. Defaults to ttnn.DRAM_MEMORY_CONFIG
        * :attr:`dtype` (Optional[ttnn.DataType]): the data type of the output tensor. Defaults to None
        * :attr:`core_grid` (Optional[Tuple[int, int]]): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to None
        * :attr:`activation` (Optional[str]): the activation function to be applied. Defaults to None

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
        activation=activation,
    )


ttnn.Tensor.__matmul__ = matmul


__all__ = []
