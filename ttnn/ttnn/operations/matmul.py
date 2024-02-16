# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple


import tt_lib as ttl

import ttnn


def _shape_is_broadcastable(input_shape_a, input_shape_b):
    if len(input_shape_a) == 1:
        batch_shape_a = []
    else:
        *batch_shape_a, _, _ = input_shape_a

    if len(input_shape_b) == 1:
        batch_shape_b = []
    else:
        *batch_shape_b, _, _ = input_shape_b

    # if width_a != height_b:
    #     return False

    len_diff = len(batch_shape_a) - len(batch_shape_b)
    if len_diff > 0:
        batch_shape_b = [1] * len_diff + batch_shape_b
    else:
        batch_shape_a = [1] * -len_diff + batch_shape_a

    return all(x == y or (x == 1 and y != 1) or (x != 1 and y == 1) for x, y in zip(batch_shape_a, batch_shape_b))


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

    if not isinstance(input_tensor_a, ttnn.Tensor):
        raise RuntimeError("Expected first argument to be a ttnn.Tensor")
    if not isinstance(input_tensor_b, ttnn.Tensor):
        raise RuntimeError("Expected second argument to be a ttnn.Tensor")

    if input_tensor_a.value.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if input_tensor_b.value.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_b must be on device!")

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

    if width_a != height_b:
        raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")

    m_size = height_a
    k_size = width_a
    n_size = width_b

    if core_grid != None:
        if m_size % ttnn.TILE_SIZE != 0 or k_size % ttnn.TILE_SIZE != 0:
            raise TypeError("The last two dimensions of the first tensor must be a multiple of 32")

        if k_size % ttnn.TILE_SIZE != 0 or n_size % ttnn.TILE_SIZE != 0:
            raise TypeError("The last two dimensions of the second tensor must be a multiple of 32")

        batch_size = math.prod(batch_shape_a)
        is_batched = math.prod(batch_shape_b) > 1

        input_tensor_a_memory_config = ttnn.get_memory_config(input_tensor_a)
        input_tensor_b_memory_config = ttnn.get_memory_config(input_tensor_b)

        if is_batched:
            if (not ttnn.is_sharded(input_tensor_a)) and (not ttnn.is_sharded(input_tensor_b)):
                per_core_M = int(math.ceil((m_size / ttnn.TILE_SIZE)))
                per_core_N = int(math.ceil((n_size / ttnn.TILE_SIZE)))
                in0_block_w = 1  # TODO(arakhmati): Can it be more than 1 without running out of memory?
            elif ttnn.is_sharded(input_tensor_a):
                if input_tensor_a_memory_config.memory_layout == ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                    raise TypeError("Cannot be width sharded")
                shard_shape = input_tensor_a_memory_config.shard_spec.shape
                N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
                per_core_M = shard_shape[0] // ttnn.TILE_SIZE
                per_core_N = N
                in0_block_w = shard_shape[1] // ttnn.TILE_SIZE
            elif ttnn.is_sharded(input_tensor_b):
                if input_tensor_b_memory_config.memory_layout == ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                    raise TypeError("Cannot be width sharded")
                shard_shape = input_tensor_b_memory_config.shard_spec.shape
                per_core_M = int(math.ceil((m_size / ttnn.TILE_SIZE)))
                per_core_N = shard_shape[1] // ttnn.TILE_SIZE
                in0_block_w = 1
        else:
            if not ttnn.is_sharded(input_tensor_a):
                per_core_M = int(math.ceil(((batch_size * m_size) / ttnn.TILE_SIZE) / core_grid[0]))
                per_core_N = int(math.ceil(n_size / ttnn.TILE_SIZE / core_grid[1]))
                in0_block_w = 4  # TODO(arakhmati): What is a good starting point?
                while (k_size // ttnn.TILE_SIZE) % in0_block_w != 0:
                    in0_block_w -= 1
            else:
                if not input_tensor_a_memory_config.memory_layout == ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED:
                    raise TypeError("Must be block sharded")
                K = input_tensor_a.shape[-1] // ttnn.TILE_SIZE
                N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
                shard_shape = input_tensor_a_memory_config.shard_spec.shape
                per_core_M = shard_shape[0] // ttnn.TILE_SIZE
                per_core_N = (N * shard_shape[1]) // (K * ttnn.TILE_SIZE)
                in0_block_w = 1

        subblocks = [
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
        for out_subblock_h, out_subblock_w in subblocks:
            if per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0:
                break

        # logger.info(
        #     f"is_batched={is_batched}, per_core_M={per_core_M}, per_core_N={per_core_N}, in0_block_w={in0_block_w}, out_subblock_h={out_subblock_h}, out_subblock_w={out_subblock_w}"
        # )

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value
        if is_batched:
            ttl_output_tensor = ttl.operations.primary.matmul(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                program_config=ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(core_grid[1], core_grid[0]),
                    in0_block_w=in0_block_w,  # k
                    out_subblock_h=out_subblock_h,  # m
                    out_subblock_w=out_subblock_w,  # n
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                ),
                output_mem_config=memory_config,
                output_dtype=dtype,
            )
        else:
            try:
                ttl_output_tensor = ttl.operations.primary.matmul(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    program_config=ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(core_grid[1], core_grid[0]),
                        in0_block_w=in0_block_w,  # k
                        out_subblock_h=out_subblock_h,  # m
                        out_subblock_w=out_subblock_w,  # n
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        transpose_mcast=False,
                        fused_activation=None,
                    ),
                    output_mem_config=memory_config,
                    output_dtype=dtype,
                )
            except Exception as e:
                raise RuntimeError(f"ttnn.matmul: ttl.operations.primary.matmul failed. {e}")

        output_tensor = ttnn.Tensor(ttl_output_tensor)

    elif height_a == 1 and width_b == 1:  # dot product
        if dtype != input_tensor_a.dtype:
            raise RuntimeError("dtype is not supported for dot product")

        ttl_input_tensor_a = input_tensor_a.value
        ttl_input_tensor_b = input_tensor_b.value

        # return a dot product
        ttl_output_tensor = ttl.tensor.bcast(
            ttl_input_tensor_a,
            ttl_input_tensor_b,
            ttl.tensor.BcastOpMath.MUL,
            ttl.tensor.BcastOpDim.H,
            output_mem_config=memory_config,
        )
        ttl_output_tensor = ttl.tensor.reduce(
            ttl_output_tensor,
            ttl.tensor.ReduceOpMath.SUM,
            ttl.tensor.ReduceOpDim.W,
            1.0,
            output_mem_config=memory_config,
        )
        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_shape = (32,)

    elif _shape_is_broadcastable(input_shape_a, input_shape_b):
        if dtype != input_tensor_a.dtype:
            raise RuntimeError("dtype is not supported for matmul without core grid")
        if width_a != height_b:
            raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")
        if all(x == 1 for x in batch_shape_b):
            ttl_input_tensor_a = input_tensor_a.value
            ttl_input_tensor_b = input_tensor_b.value
            output_tensor = ttnn.Tensor(
                ttl.tensor.matmul(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)
            )
        else:
            ttl_input_tensor_a = input_tensor_a.value
            ttl_input_tensor_b = input_tensor_b.value
            output_tensor = ttnn.Tensor(
                ttl.tensor.bmm(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)
            )

    else:
        raise RuntimeError("These tensors cannot be broadcasted")

    if output_tensor.shape != output_shape:
        output_tensor = ttnn.reshape(output_tensor, output_shape)
    return output_tensor


ttnn.Tensor.__matmul__ = matmul


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
        bias = ttnn.to_torch(bias)
        if len(bias.shape) == 2:
            bias = bias[0]
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
        raise RuntimeError("There is currently no support for ranks greater than 4.")

    if len(input_shape_b) > 4:
        raise RuntimeError(f"There is currently no support for ranks greater than {4}.")

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
        raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")

    m_size = height_a
    k_size = width_a
    n_size = width_b

    ttl_input_tensor_a = input_tensor_a.value
    ttl_input_tensor_b = input_tensor_b.value

    if core_grid != None:
        if m_size % ttnn.TILE_SIZE != 0 or k_size % ttnn.TILE_SIZE != 0:
            raise TypeError("The last two dimensions of the first tensor must be a multiple of 32")

        if k_size % ttnn.TILE_SIZE != 0 or n_size % ttnn.TILE_SIZE != 0:
            raise TypeError("The last two dimensions of the second tensor must be a multiple of 32")

        batch_size = math.prod(batch_shape_a)
        is_batched = math.prod(batch_shape_b) > 1

        if is_batched:
            per_core_M = int(math.ceil((m_size / ttnn.TILE_SIZE)))
            per_core_N = int(math.ceil((n_size / ttnn.TILE_SIZE)))
            in0_block_w = 1  # TODO(arakhmati): Can it be more than 1 without running out of memory?
        else:
            per_core_M = int(math.ceil(((batch_size * m_size) / ttnn.TILE_SIZE) / core_grid[0]))
            per_core_N = int(math.ceil(n_size / ttnn.TILE_SIZE / core_grid[1]))
            in0_block_w = 4  # TODO(arakhmati): What is a good starting point?
            while (k_size // ttnn.TILE_SIZE) % in0_block_w != 0:
                in0_block_w -= 1

        subblocks = [
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
        for out_subblock_h, out_subblock_w in subblocks:
            if per_core_M % out_subblock_h == 0 and per_core_N % out_subblock_w == 0:
                break

        # logger.info(
        #     f"is_batched={is_batched}, per_core_M={per_core_M}, per_core_N={per_core_N}, in0_block_w={in0_block_w}, out_subblock_h={out_subblock_h}, out_subblock_w={out_subblock_w}"
        # )
        if is_batched:
            if bias is not None:
                raise RuntimeError("bias must be None")
            if activation is not None:
                raise RuntimeError("activations must be None")
            ttl_output_tensor = ttl.operations.primary.matmul(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                program_config=ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(core_grid[1], core_grid[0]),
                    in0_block_w=in0_block_w,  # k
                    out_subblock_h=out_subblock_h,  # m
                    out_subblock_w=out_subblock_w,  # n
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                ),
                output_mem_config=memory_config,
                output_dtype=dtype,
            )
        else:
            ttl_bias = bias.value if bias is not None else None
            if activation == "gelu":
                fused_activation = (ttl.tensor.FusibleActivation.GELU, True)
            elif activation == "relu":
                fused_activation = ttl.tensor.FusibleActivation.RELU
            elif activation is None:
                fused_activation = None
            else:
                raise RuntimeError(f"{activation} is not supported as activation function")

            try:
                ttl_output_tensor = ttl.operations.primary.matmul(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    bias=ttl_bias,
                    program_config=ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(core_grid[1], core_grid[0]),
                        in0_block_w=in0_block_w,  # k
                        out_subblock_h=out_subblock_h,  # m
                        out_subblock_w=out_subblock_w,  # n
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        transpose_mcast=False,
                        fused_activation=fused_activation,
                    ),
                    output_mem_config=memory_config,
                    output_dtype=dtype,
                )
            except Exception as e:
                raise RuntimeError(f"ttnn.linear: ttl.operations.primary.matmul failed. {e}")

        output_tensor = ttnn.Tensor(ttl_output_tensor)

    else:
        if activation is not None:
            raise RuntimeError("activation must be None")
        ttl_bias = bias.value if bias is not None else None
        ttl_output_tensor = ttl.operations.primary.matmul(
            ttl_input_tensor_a,
            ttl_input_tensor_b,
            bias=ttl_bias,
            output_mem_config=memory_config,
            output_dtype=dtype,
        )

        output_tensor = ttnn.Tensor(ttl_output_tensor)

    if output_tensor.shape != output_shape:
        output_tensor = ttnn.reshape(output_tensor, output_shape)
    return output_tensor


__all__ = []
