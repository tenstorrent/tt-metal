# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pathlib
from typing import Optional, Tuple, Union


import tt_lib as ttl

from ttnn.tensor import (
    Shape,
    Tensor,
    from_torch,
    to_torch,
    to_device,
    from_device,
    to_layout,
    DataType,
    MemoryConfig,
    DRAM_MEMORY_CONFIG,
    Layout,
    ROW_MAJOR_LAYOUT,
    TILE_LAYOUT,
    TILE_SIZE,
    has_storage_type_of,
)
from ttnn.decorators import decorate_operation

MODEL_CACHE_PATH = pathlib.Path().home() / ".cache" / "tenstorrent"


MAX_RANK = 4

DEVICES = {}


def open(device_id: int):
    if device_id in DEVICES:
        return DEVICES[device_id]
    device = ttl.device.CreateDevice(device_id)
    DEVICES[device_id] = device
    return device


def close(device):
    ttl.device.CloseDevice(device)
    del DEVICES[device.id()]


def enable_program_cache():
    ttl.program_cache.enable()


def _is_scalar(value):
    return isinstance(value, (int, float, complex))


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


# TODO(arakhmati): remove this once underlying C++ code can handle non-4D shapes
def _reshape_to_4D(tensor):
    if len(tensor.shape) == 4:
        return tensor
    if len(tensor.shape) > 4:
        raise RuntimeError("Tensor cannot have more than 4 dimensions!")
    num_missing_dims = 4 - len(tensor.shape)
    shape = tuple(tensor.shape)
    full_shape = tuple(tensor.shape.padded())
    shape = (1,) * num_missing_dims + shape
    full_shape = (1,) * num_missing_dims + full_shape
    return reshape(tensor, shape=Shape(shape, full_shape))


# Math Operations


def _torch_matmul(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
    input_tensor_a = from_device(input_tensor_a)
    input_tensor_a = to_layout(input_tensor_a, ROW_MAJOR_LAYOUT)
    input_tensor_a = to_torch(input_tensor_a)

    input_tensor_b = from_device(input_tensor_b)
    input_tensor_b = to_layout(input_tensor_b, ROW_MAJOR_LAYOUT)
    input_tensor_b = to_torch(input_tensor_b)

    return input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)


@decorate_operation(torch_function=_torch_matmul)
def matmul(
    input_tensor_a: Tensor,
    input_tensor_b: Tensor,
    *,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    dtype: Optional[DataType] = None,
    core_grid: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """
    matmul(input_tensor_a: Tensor, input_tensor_b: Tensor, *, memory_config: MemoryConfig=DRAM_MEMORY_CONFIG, core_grid: Optional[Tuple[int, int]] = None) -> Tensor

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

      Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
      are broadcastable, and not the matrix dimensions. For example, if :attr:`input_tensor_a` is a
      :math:`(j \\times 1 \\times n \\times m)` tensor and :attr:`input_tensor_b` is a :math:`(k \\times m \\times p)`
      tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
      matrix dimensions) are different. The operation will return a :math:`(j \\times k \\times n \\times p)` tensor.


    .. note::

        The 1-dimensional dot product version of this function is currently returning the Tensor with a non-empty shape. This is expected to be fixed in an upcomming release.

    Arguments:
        * :attr:`input_tensor_a` (Tensor): the first tensor to be multiplied
        * :attr:`input_tensor_b` (Tensor): the second tensor to be multiplied

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
        padded_output_shape_list.append(input_shape_a.padded()[index])
    output_shape_list.append(input_shape_b[-1])
    padded_output_shape_list.append(input_shape_b.padded()[-1])
    output_shape = Shape(output_shape_list, padded_output_shape_list)

    if not isinstance(input_tensor_a, Tensor):
        raise RuntimeError("Expected first argument to be a ttnn.Tensor")
    if not isinstance(input_tensor_b, Tensor):
        raise RuntimeError("Expected second argument to be a ttnn.Tensor")

    if input_tensor_a.value.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if input_tensor_b.value.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_b must be on device!")

    # The idea is to make the shapes "possibly" broadcastable.
    if len(input_tensor_a.shape) > MAX_RANK:
        raise RuntimeError("There is currently no support for ranks greater than 4.")

    if len(input_shape_b) > MAX_RANK:
        raise RuntimeError(f"There is currently no support for ranks greater than {MAX_RANK}.")

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

    input_tensor_a = _reshape_to_4D(input_tensor_a)
    input_tensor_b = _reshape_to_4D(input_tensor_b)

    if width_a != height_b:
        raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")

    m_size = height_a
    k_size = width_a
    n_size = width_b

    if core_grid != None:
        if m_size % TILE_SIZE != 0 or k_size % TILE_SIZE != 0:
            raise TypeError("The last two dimensions of the first tensor must be a multiple of 32")

        if k_size % TILE_SIZE != 0 or n_size % TILE_SIZE != 0:
            raise TypeError("The last two dimensions of the second tensor must be a multiple of 32")

        batch_size = math.prod(batch_shape_a)
        is_batched = math.prod(batch_shape_b) > 1

        if is_batched:
            per_core_M = int(math.ceil((m_size / TILE_SIZE)))
            per_core_N = int(math.ceil((n_size / TILE_SIZE)))
            in0_block_w = 1  # TODO(arakhmati): Can it be more than 1 without running out of memory?
        else:
            per_core_M = int(math.ceil(((batch_size * m_size) / TILE_SIZE) / core_grid[0]))
            per_core_N = int(math.ceil(n_size / TILE_SIZE / core_grid[1]))
            in0_block_w = 4  # TODO(arakhmati): What is a good starting point?
            while (k_size // TILE_SIZE) % in0_block_w != 0:
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

        output_tensor = Tensor(ttl_output_tensor)

    elif height_a == 1 and width_b == 1:  # dot product
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
        output_tensor = Tensor(ttl_output_tensor)
        output_shape = (32,)

    elif _shape_is_broadcastable(input_shape_a, input_shape_b):
        if width_a != height_b:
            raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")
        if all(x == 1 for x in batch_shape_b):
            ttl_input_tensor_a = input_tensor_a.value
            ttl_input_tensor_b = input_tensor_b.value
            output_tensor = Tensor(
                ttl.tensor.matmul(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)
            )
        else:
            ttl_input_tensor_a = input_tensor_a.value
            ttl_input_tensor_b = input_tensor_b.value
            output_tensor = Tensor(
                ttl.tensor.bmm(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)
            )

    else:
        raise RuntimeError("These tensors cannot be broadcasted")

    if output_tensor.shape != output_shape:
        output_tensor = reshape(output_tensor, output_shape)
    return output_tensor


def _torch_linear(input_tensor_a: Tensor, input_tensor_b: Tensor, *, bias=None, activation=None, **_):
    import torch

    input_tensor_a = from_device(input_tensor_a)
    input_tensor_a = to_layout(input_tensor_a, ROW_MAJOR_LAYOUT)
    input_tensor_a = to_torch(input_tensor_a)

    input_tensor_b = from_device(input_tensor_b)
    input_tensor_b = to_layout(input_tensor_b, ROW_MAJOR_LAYOUT)
    input_tensor_b = to_torch(input_tensor_b)

    output_tensor = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)

    if bias is not None:
        bias = from_device(bias)
        bias = to_layout(bias, ROW_MAJOR_LAYOUT)
        bias = to_torch(bias)
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


@decorate_operation(torch_function=_torch_linear)
def linear(
    input_tensor_a: Tensor,
    input_tensor_b: Tensor,
    *,
    bias: Optional[Tensor] = None,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    dtype: Optional[DataType] = None,
    core_grid: Optional[Tuple[int, int]] = None,
    activation: Optional[str] = None,
) -> Tensor:
    """
    linear(input_tensor_a: Tensor, input_tensor_b: Tensor, *, bias: Optional[Tensor] = None, memory_config: MemoryConfig=DRAM_MEMORY_CONFIG, dtype: Optional[DataType] = None, core_grid: Optional[Tuple[int, int]] = None, activation: Optional[str] = None) -> Tensor

    Returns the linear transformation of the inputs

    Arguments:
        * :attr:`input_tensor_a` (Tensor): the first tensor to be multiplied
        * :attr:`input_tensor_b` (Tensor): the second tensor to be multiplied

    Example::
        >>> # batched matrix x broadcasted matrix
        >>> activations = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> weight = ttnn.to_device(ttnn.from_torch(torch.randn((32, 128), dtype=torch.bfloat16)), device)
        >>> bias = ttnn.to_device(ttnn.from_torch(torch.randn((128,), dtype=torch.bfloat16)), device)
        >>> output = torch.linear(activations, weight, bias=bias)
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
        padded_output_shape_list.append(input_shape_a.padded()[index])
    output_shape_list.append(input_shape_b[-1])
    padded_output_shape_list.append(input_shape_b.padded()[-1])
    output_shape = Shape(output_shape_list, padded_output_shape_list)

    if not isinstance(input_tensor_a, Tensor):
        raise RuntimeError("Expected first argument to be a ttnn.Tensor")
    if not isinstance(input_tensor_b, Tensor):
        raise RuntimeError("Expected second argument to be a ttnn.Tensor")

    if input_tensor_a.value.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if input_tensor_b.value.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_b must be on device!")

    # The idea is to make the shapes "possibly" broadcastable.
    if len(input_tensor_a.shape) > MAX_RANK:
        raise RuntimeError("There is currently no support for ranks greater than 4.")

    if len(input_shape_b) > MAX_RANK:
        raise RuntimeError(f"There is currently no support for ranks greater than {MAX_RANK}.")

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
    input_tensor_a = _reshape_to_4D(input_tensor_a)
    input_tensor_b = _reshape_to_4D(input_tensor_b)

    if bias is not None:
        bias = _reshape_to_4D(bias)

    if width_a != height_b:
        raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")

    m_size = height_a
    k_size = width_a
    n_size = width_b

    ttl_input_tensor_a = input_tensor_a.value
    ttl_input_tensor_b = input_tensor_b.value

    if core_grid != None:
        if m_size % TILE_SIZE != 0 or k_size % TILE_SIZE != 0:
            raise TypeError("The last two dimensions of the first tensor must be a multiple of 32")

        if k_size % TILE_SIZE != 0 or n_size % TILE_SIZE != 0:
            raise TypeError("The last two dimensions of the second tensor must be a multiple of 32")

        batch_size = math.prod(batch_shape_a)
        is_batched = math.prod(batch_shape_b) > 1

        if is_batched:
            per_core_M = int(math.ceil((m_size / TILE_SIZE)))
            per_core_N = int(math.ceil((n_size / TILE_SIZE)))
            in0_block_w = 1  # TODO(arakhmati): Can it be more than 1 without running out of memory?
        else:
            per_core_M = int(math.ceil(((batch_size * m_size) / TILE_SIZE) / core_grid[0]))
            per_core_N = int(math.ceil(n_size / TILE_SIZE / core_grid[1]))
            in0_block_w = 4  # TODO(arakhmati): What is a good starting point?
            while (k_size // TILE_SIZE) % in0_block_w != 0:
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

        output_tensor = Tensor(ttl_output_tensor)

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

        output_tensor = Tensor(ttl_output_tensor)

    if output_tensor.shape != output_shape:
        output_tensor = reshape(output_tensor, output_shape)
    return output_tensor


def _torch_add(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
    input_shape_a = input_tensor_a.shape
    slices = [slice(0, dim) for dim in input_shape_a]
    input_tensor_a = from_device(input_tensor_a)
    input_tensor_a = to_layout(input_tensor_a, ROW_MAJOR_LAYOUT)
    input_tensor_a = to_torch(input_tensor_a)
    input_tensor_a = input_tensor_a[slices]

    if not _is_scalar(input_tensor_b):
        input_shape_b = input_tensor_b.shape
        slices = [slice(0, dim) for dim in input_shape_b]
        input_tensor_b = from_device(input_tensor_b)
        input_tensor_b = to_layout(input_tensor_b, ROW_MAJOR_LAYOUT)
        input_tensor_b = to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b[slices]

    return input_tensor_a + input_tensor_b


@decorate_operation(torch_function=_torch_add)
def add(
    input_tensor_a: Tensor,
    input_tensor_b: Union[Tensor, int, float],
    *,
    alpha: Union[int, float] = 1,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
) -> Tensor:
    r"""
    add(input_tensor_a: Tensor, input_tensor_b: Union[Tensor, int, float], *, alpha: Union[int, float]=1) -> Tensor

    Adds :attr:`input_tensor_b`, scaled by :attr:`alpha`, to :attr:`input_tensor_a` and returns the tensor with the same layout as :attr:`input_tensor_a`

    .. math::
        \mathrm{{input\_tensor\_a}}_i + \mathrm{{alpha}} \\times \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

    Keyword args:
        :attr:`alpha` (Number): the multiplier for :attr:`input_tensor_b`.

    Example::

        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
        >>> output = ttnn.add(tensor1, tensor2, alpha=2)
        >>> print(output)
        Tensor([ 1, 4], dtype=bfloat16 )

    """

    if not isinstance(input_tensor_a, Tensor):
        raise TypeError("Expected first argument to be a ttnn.Tensor")

    # Swap tensors if input_tensor_a needs to be broadcasted to input_tensor_b
    if (
        isinstance(input_tensor_a, Tensor)
        and isinstance(input_tensor_b, Tensor)
        and math.prod(input_tensor_a.shape) < math.prod(input_tensor_b.shape)
    ):
        input_tensor_a, input_tensor_b = input_tensor_b, input_tensor_a

    original_shape = input_tensor_a.shape
    input_tensor_a = _reshape_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a.value

    if not has_storage_type_of(input_tensor_a, ttl.tensor.StorageType.DEVICE):
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        output_tensor = Tensor(
            ttl.tensor.add_unary(
                ttl_input_tensor_a,
                input_tensor_b * alpha,
                output_mem_config=memory_config,
            )
        )
        return reshape(output_tensor, original_shape)
    elif isinstance(input_tensor_b, Tensor):
        input_shape_b = input_tensor_b.shape

        if len(input_shape_b) == 1:
            height_b = 1
            (width_b,) = input_shape_b
        else:
            *_, height_b, width_b = input_shape_b

        input_tensor_b = _reshape_to_4D(input_tensor_b)
        ttl_input_tensor_b = input_tensor_b.value
        if ttl_input_tensor_b.storage_type() != ttl.tensor.StorageType.DEVICE:
            raise RuntimeError("input_tensor_a must be on device!")
    else:
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    ttl_input_tensor_b = input_tensor_b.value
    if alpha != 1:
        ttl_input_tensor_b = ttl.tensor.mul_unary(
            ttl_input_tensor_b,
            alpha,
            output_mem_config=memory_config,
        )

    if height_b == 1 and width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.HW,
                output_mem_config=memory_config,
            )
        )
    elif height_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.H,
                output_mem_config=memory_config,
            )
        )
    elif width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.W,
                output_mem_config=memory_config,
            )
        )
    else:
        output_tensor = Tensor(
            ttl.tensor.add(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                output_mem_config=memory_config,
            )
        )

    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor


def _torch_sub(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
    input_shape_a = input_tensor_a.shape
    slices = [slice(0, dim) for dim in input_shape_a]
    input_tensor_a = from_device(input_tensor_a)
    input_tensor_a = to_layout(input_tensor_a, ROW_MAJOR_LAYOUT)
    input_tensor_a = to_torch(input_tensor_a)
    input_tensor_a = input_tensor_a[slices]

    if not _is_scalar(input_tensor_b):
        input_shape_b = input_tensor_b.shape
        slices = [slice(0, dim) for dim in input_shape_b]
        input_tensor_b = from_device(input_tensor_b)
        input_tensor_b = to_layout(input_tensor_b, ROW_MAJOR_LAYOUT)
        input_tensor_b = to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b[slices]

    return input_tensor_a - input_tensor_b


@decorate_operation(torch_function=_torch_sub)
def sub(
    input_tensor_a: Tensor,
    input_tensor_b: Union[Tensor, int, float],
    *,
    alpha: Union[int, float] = 1,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
) -> Tensor:
    r"""
    sub(input_tensor_a: Tensor, input_tensor_b: Union[Tensor, int, float], *, alpha: Union[int, float]=1) -> Tensor:

    Subtracts :attr:`input_tensor_b`, scaled by :attr:`alpha`, from :attr:`input_tensor_a`.

    .. math::
        \mathrm{{input\_tensor\_a}}_i - \mathrm{{alpha}} \\times \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (Tensor or Number): the tensor or number to subtract from :attr:`input_tensor_a`.

    Keyword args:
        :attr:`alpha` (Number): the multiplier for :attr:`input_tensor_b`.

    Example::

        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
        >>> output = ttnn.sub(tensor1, tensor2, alpha=2)
        >>> print(output)
        Tensor([ 1, 0], dtype=bfloat16 )
    """
    if not isinstance(input_tensor_a, Tensor):
        raise TypeError("Expected first argument to be a ttnn.Tensor")

    original_shape = tuple(input_tensor_a.shape)
    input_tensor_a = _reshape_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a.value

    if ttl_input_tensor_a.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        output_tensor = Tensor(
            ttl.tensor.sub_unary(
                ttl_input_tensor_a,
                input_tensor_b * alpha,
                output_mem_config=memory_config,
            )
        )
        return reshape(output_tensor, original_shape)
    elif isinstance(input_tensor_b, Tensor):
        input_shape_b = input_tensor_b.shape

        if len(input_shape_b) == 1:
            height_b = 1
            (width_b,) = input_shape_b
        else:
            *_, height_b, width_b = input_shape_b

        input_tensor_b = _reshape_to_4D(input_tensor_b)
        ttl_input_tensor_b = input_tensor_b.value
        if ttl_input_tensor_b.storage_type() != ttl.tensor.StorageType.DEVICE:
            raise RuntimeError("input_tensor_a must be on device!")
    else:
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    ttl_input_tensor_b = input_tensor_b.value

    if alpha != 1:
        ttl_input_tensor_b = ttl.tensor.mul_unary(
            ttl_input_tensor_b,
            alpha,
            output_mem_config=memory_config,
        )

    if height_b == 1 and width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.SUB,
                ttl.tensor.BcastOpDim.HW,
                output_mem_config=memory_config,
            )
        )
    elif height_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.SUB,
                ttl.tensor.BcastOpDim.H,
                output_mem_config=memory_config,
            )
        )
    elif width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                ttl.tensor.BcastOpMath.SUB,
                ttl.tensor.BcastOpDim.W,
                output_mem_config=memory_config,
            )
        )
    else:
        output_tensor = Tensor(
            ttl.tensor.sub(
                ttl_input_tensor_a,
                ttl_input_tensor_b,
                output_mem_config=memory_config,
            )
        )

    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor


def _torch_mul(input_tensor_a: Tensor, input_tensor_b: Tensor, **_):
    input_shape_a = input_tensor_a.shape
    slices = [slice(0, dim) for dim in input_shape_a]
    input_tensor_a = from_device(input_tensor_a)
    input_tensor_a = to_layout(input_tensor_a, ROW_MAJOR_LAYOUT)
    input_tensor_a = to_torch(input_tensor_a)
    input_tensor_a = input_tensor_a[slices]

    if not _is_scalar(input_tensor_b):
        input_shape_b = input_tensor_b.shape
        slices = [slice(0, dim) for dim in input_shape_b]
        input_tensor_b = from_device(input_tensor_b)
        input_tensor_b = to_layout(input_tensor_b, ROW_MAJOR_LAYOUT)
        input_tensor_b = to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b[slices]

    return input_tensor_a * input_tensor_b


@decorate_operation(torch_function=_torch_mul)
def mul(input_tensor_a: Tensor, input_tensor_b: Tensor, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor:
    r"""
    mul(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor

    Multiples :attr:`input_tensor_a` and :attr:`input_tensor_b` element-wise.

    .. math::
        \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (Tensor or Number): the tensor or number to multiply with :attr:`input_tensor_a`.

    Example::

        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
        >>> output = ttnn.mul(tensor1, tensor2)
        >>> print(output)
        Tensor([ 0, 2], dtype=bfloat16 )

    """

    original_shape = input_tensor_a.shape
    input_tensor_a = _reshape_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a.value

    if not isinstance(input_tensor_a, Tensor):
        raise TypeError("Expected first argument to be a ttnn.Tensor")

    ttl_input_tensor_a = input_tensor_a.value

    if not has_storage_type_of(input_tensor_a, ttl.tensor.StorageType.DEVICE):
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        return reshape(
            Tensor(
                ttl.tensor.mul_unary(
                    ttl_input_tensor_a,
                    input_tensor_b,
                    output_mem_config=memory_config,
                )
            ),
            original_shape,
        )
    elif not isinstance(input_tensor_b, Tensor):
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    input_shape_b = input_tensor_b.shape

    if len(input_shape_b) == 1:
        height_b = 1
        (width_b,) = input_shape_b
    else:
        *_, height_b, width_b = input_shape_b

    input_tensor_b = _reshape_to_4D(input_tensor_b)
    ttl_input_tensor_b = input_tensor_b.value

    if height_b == 1 and width_b == 1:
        return reshape(
            Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    ttl.tensor.BcastOpMath.MUL,
                    ttl.tensor.BcastOpDim.HW,
                    output_mem_config=memory_config,
                ),
                original_shape,
            )
        )
    elif height_b == 1:
        return reshape(
            Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    ttl.tensor.BcastOpMath.MUL,
                    ttl.tensor.BcastOpDim.H,
                    output_mem_config=memory_config,
                )
            ),
            original_shape,
        )
    elif width_b == 1:
        return reshape(
            Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    ttl.tensor.BcastOpMath.MUL,
                    ttl.tensor.BcastOpDim.W,
                    output_mem_config=memory_config,
                )
            ),
            original_shape,
        )

    return reshape(
        Tensor(ttl.tensor.mul(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)),
        original_shape,
    )


subtract = sub
multiply = mul


Tensor.__matmul__ = matmul
Tensor.__add__ = add
Tensor.__radd__ = add
Tensor.__sub__ = sub
Tensor.__mul__ = mul
Tensor.__rmul__ = mul


# Data Transformations
def _torch_reshape(input_tensor: Tensor, shape: Union[Shape, Tuple[int, ...]], **_):
    import torch
    import ttnn

    input_tensor = from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = to_torch(input_tensor)

    if isinstance(shape, Shape):
        shape = tuple(shape.padded())

    return torch.reshape(input_tensor, shape).contiguous().clone()


@decorate_operation(torch_function=_torch_reshape)
def reshape(input_tensor: Tensor, shape: Union[Shape, Tuple[int, ...]]) -> Tensor:
    r"""
    reshape(input_tensor: Tensor, shape: Union[Shape, Tuple[int, ...]]) -> Tensor

    Reshape :attr:`input_tensor` into :attr:`shape`.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`shape`: the desired shape.

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.reshape(tensor, (32, 64))
        >>> print(output.shape)
        ttnn.Shape([32, 64])

    """

    if isinstance(shape, tuple):
        if not (0 <= shape.count(-1) <= 1):
            raise RuntimeError("Shape cannot have more than 1 elements that is set to -1!")

        volume = math.prod(input_tensor.shape)
        new_volume = math.prod(shape)
        if new_volume < 0:
            index_of_negative_1 = shape.index(-1)
            shape = list(shape)
            shape[index_of_negative_1] = volume // (-new_volume)
            shape = tuple(shape)
        shape = Shape(shape)

    if not isinstance(shape, Shape):
        raise RuntimeError("Shape must be of type Shape")

    if input_tensor.shape == shape:
        return input_tensor

    def ttnn_reshape(tensor, shape):
        ttl_input_tensor = tensor.value
        return Tensor(ttl_input_tensor.reshape(shape.value))

    ttnn_reshape = ttl.tensor.decorate_external_operation(ttnn_reshape, function_name="ttnn.reshape")

    if input_tensor.is_contiguous():
        if has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            # Page size depends on the width, so only modify the shape if the width is the same
            if input_tensor.shape[-1] == shape[-1]:
                return ttnn_reshape(input_tensor, shape)
        else:
            return ttnn_reshape(input_tensor, shape)

    if input_tensor.layout == TILE_LAYOUT:
        *_, new_height, new_width = tuple(shape.padded())
        if new_height % TILE_SIZE == 0 and new_width % TILE_SIZE == 0:
            return ttnn_reshape(input_tensor, shape)

    if (
        has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE)
        and len(input_tensor.shape) == 4
        and len(shape) == 4
    ):
        ttl_input_tensor = input_tensor.value
        w, z, y, x = shape
        ttl_output_tensor = ttl.tensor.reshape(ttl_input_tensor, w, z, y, x)
        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = ttnn_reshape(output_tensor, shape)
        return output_tensor
    else:

        def torch_reshape(tensor, shape):
            return tensor.reshape(tuple(shape.padded())).contiguous().clone()

        if has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            ttl_input_tensor = input_tensor.value
            device = ttl_input_tensor.device()
            tensor = to_layout(input_tensor, ROW_MAJOR_LAYOUT)
            tensor = from_device(tensor)
            tensor = to_torch(tensor)
            tensor = ttl.tensor.decorate_external_operation(torch_reshape, function_name="torch.reshape")(tensor, shape)
            tensor = from_torch(tensor, input_tensor.dtype)
            tensor = to_device(tensor, device)
            tensor = ttnn_reshape(tensor, shape)
        else:
            tensor = to_layout(input_tensor, ROW_MAJOR_LAYOUT)
            tensor = to_torch(tensor)
            tensor = ttl.tensor.decorate_external_operation(torch_reshape, function_name="torch.reshape")(tensor, shape)
            tensor = from_torch(tensor, input_tensor.dtype)
            tensor = ttnn_reshape(tensor, shape)

        return tensor


def _torch_permute(input_tensor: Tensor, order: Tuple[int, ...], **_):
    import torch
    import ttnn

    input_tensor = from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = to_torch(input_tensor)

    return torch.permute(input_tensor, order).contiguous().clone()


@decorate_operation(torch_function=_torch_permute)
def permute(input_tensor: Tensor, order: Tuple[int, ...]) -> Tensor:
    r"""
    permute(input_tensor: Tensor, order: Tuple[int, ...]) -> Tensor

    Permutes :attr:`input_tensor` using :attr:`order`.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`order`: the desired ordering of dimensions.

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
        >>> print(output.shape)
        [1, 1, 32, 64]

    """

    if not isinstance(order, tuple):
        raise RuntimeError("order must be a tuple")

    if not has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
        RuntimeError("input_tensor must be on device!")

    ttl_input_tensor = input_tensor.value

    if len(input_tensor.shape) != len(order):
        raise RuntimeError(
            "The number of dimensions in the tensor input does not match the length of the desired ordering"
        )

    if has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE) and len(input_tensor.shape) == 4:
        return Tensor(ttl.tensor.permute(ttl_input_tensor, order))
    elif len(input_tensor.shape) < 4:
        original_shape = tuple(input_tensor.shape)
        desired_shape = tuple([original_shape[i] for i in order])
        input_tensor = _reshape_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value
        new_order = order
        while len(new_order) < 4:
            new_order = (0,) + tuple(x + 1 for x in new_order)
        output_tensor = Tensor(ttl.tensor.permute(ttl_input_tensor, new_order))
        return reshape(output_tensor, desired_shape)
    else:

        def torch_permute(tensor, order):
            return tensor.permute(order).contiguous().clone()

        device = ttl_input_tensor.device()
        tensor = to_layout(input_tensor, ROW_MAJOR_LAYOUT)
        tensor = from_device(tensor)
        tensor = to_torch(tensor)
        tensor = ttl.tensor.decorate_external_operation(torch_permute, function_name="torch.permute")(tensor, order)
        tensor = from_torch(tensor, input_tensor.dtype)
        tensor = to_device(tensor, device)
        return tensor


def _torch_embedding(input_tensor: Tensor, weight: Tensor, **_):
    import torch

    input_tensor = from_device(input_tensor)
    input_tensor = to_torch(input_tensor)

    weight = from_device(weight)
    weight = to_torch(weight)

    output_tensor = torch.nn.functional.embedding(input_tensor, weight)

    return output_tensor


@decorate_operation(torch_function=_torch_embedding)
def embedding(
    input_tensor: Tensor,
    weight: Tensor,
    *,
    layout: Layout = ROW_MAJOR_LAYOUT,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
):
    r"""
    embedding(inxput_tensor: ttnn.Tensor, weight: ttnn.Tensor) -> None

    Retrieves word embeddings using input_tensor. The input_tensor is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.

    Args:
        * :attr:`input_tensor`: the indices ttnn.Tensor
        * :attr:`weight`: the embeddings ttnn.Tensor that correspond to the indices ttnn.Tensor

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> input_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device)
        >>> # an embedding matrix containing 10 tensors of size 4
        >>> weight = ttnn.to_device(ttnn.from_torch(torch.rand(10, 4), dtype=ttnn.bfloat16), device)
        >>> ttnn.embedding(input_tensor, weight)
        Tensor([ [[1, 0.106445, 0.988281, 0.59375],
            [0.212891, 0.964844, 0.199219, 0.996094],
            [3.78362e-38, 0, 7.89785e-39, 0],
            [8.04479e-38, 0, 1.25815e-38, 0]],

           [[2.71833e-38, 0, 3.59995e-38, 0],
            [7.60398e-38, 0, 1.83671e-38, 0],
            [2.22242e-38, 0, 1.88263e-38, 0],
            [1.35917e-38, 0, 4.49994e-39, 0]]], dtype=bfloat16 )

    """
    if len(input_tensor.shape) != 2:
        raise RuntimeError("Input Tensor must have rank of 2!")
    if len(weight.shape) not in {2, 4}:
        raise RuntimeError("Weight Tensor must either have rank of 2 or 4!")

    *_, hidden_embedding_dim = tuple(weight.shape)
    weight = _reshape_to_4D(weight)

    batch_size, sentence_size = input_tensor.shape
    input_tensor = reshape(input_tensor, shape=(batch_size, 1, 1, sentence_size))

    tilized = layout == TILE_LAYOUT
    embeddings = Tensor(
        ttl.tensor.embeddings(input_tensor.value, weight.value, tilized, output_mem_config=memory_config)
    )
    embeddings = reshape(embeddings, shape=(batch_size, sentence_size, hidden_embedding_dim))

    return embeddings


def _torch_softmax(input_tensor: Tensor, dim: int, **_):
    import torch

    input_tensor = from_device(input_tensor)
    input_tensor = to_layout(input_tensor, ROW_MAJOR_LAYOUT)
    input_tensor = to_torch(input_tensor)

    return torch.softmax(input_tensor, dim)


@decorate_operation(torch_function=_torch_softmax)
def softmax(input_tensor: Tensor, dim: int, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> Tensor:
    r"""
    softmax(input_tensor: Tensor, dim: int) -> Tensor

    Compute softmax over :attr:`input_tensor` along :attr:`dim`.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`dim`: the dimension along which to compute softmax.

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.softmax(tensor, -1)
        >>> print(output[0, 0, 0, :3])
        Tensor([ 0.0310059, 0.0310059, 0.0310059], dtype=bfloat16 )

    """

    input_shape = input_tensor.shape
    rank = len(input_shape)
    if dim < 0:
        dim = rank + dim

    input_tensor = _reshape_to_4D(input_tensor)

    ttl_input_tensor = input_tensor.value
    if dim == rank - 1:
        ttl_output_tensor = ttl.tensor.softmax(ttl_input_tensor, output_mem_config=memory_config)
    else:
        dim_4D = dim + 4 - rank
        ttl_output_tensor = ttl.operations.primary.moreh_softmax(
            ttl_input_tensor, dim=dim_4D, output_mem_config=memory_config
        )
    output_tensor = Tensor(ttl_output_tensor)
    output_tensor = reshape(output_tensor, input_shape)
    return output_tensor


def _torch_layer_norm(input_tensor: Tensor, *, epsilon=1e-12, residual_input_tensor=None, weight=None, bias=None, **_):
    import torch

    input_tensor = from_device(input_tensor)
    input_tensor = to_layout(input_tensor, ROW_MAJOR_LAYOUT)
    input_tensor = to_torch(input_tensor)

    if residual_input_tensor is not None:
        residual_input_tensor = from_device(residual_input_tensor)
        residual_input_tensor = to_layout(residual_input_tensor, ROW_MAJOR_LAYOUT)
        residual_input_tensor = to_torch(residual_input_tensor)
        input_tensor += residual_input_tensor

    if weight is not None:
        weight = from_device(weight)
        weight = to_layout(weight, ROW_MAJOR_LAYOUT)
        weight = to_torch(weight)
        if len(weight.shape) == 2:
            weight = weight[0]

    if bias is not None:
        bias = from_device(bias)
        bias = to_layout(bias, ROW_MAJOR_LAYOUT)
        bias = to_torch(bias)
        if len(bias.shape) == 2:
            bias = bias[0]

    return torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight, bias, eps=epsilon)


@decorate_operation(torch_function=_torch_layer_norm)
def layer_norm(
    input_tensor: Tensor,
    *,
    epsilon: float = 1e-12,
    residual_input_tensor: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
) -> Tensor:
    r"""
    layer_norm(input_tensor: Tensor) -> Tensor

    Compute layer_norm over :attr:`input_tensor`.

    """

    original_shape = input_tensor.shape
    input_tensor = _reshape_to_4D(input_tensor)
    if residual_input_tensor is not None:
        residual_input_tensor = _reshape_to_4D(residual_input_tensor)
    if weight is not None:
        weight = _reshape_to_4D(weight)
    if bias is not None:
        bias = _reshape_to_4D(bias)

    ttl_input_tensor = input_tensor.value
    residual_input_tensor = residual_input_tensor.value if residual_input_tensor is not None else None
    ttl_weight = weight.value if weight is not None else None
    ttl_bias = bias.value if bias is not None else None

    if residual_input_tensor is not None:
        output_tensor = ttl.tensor.add_layernorm(
            ttl_input_tensor, residual_input_tensor, epsilon, ttl_weight, ttl_bias, output_mem_config=memory_config
        )
    else:
        output_tensor = ttl.tensor.layernorm(
            ttl_input_tensor, epsilon, ttl_weight, ttl_bias, output_mem_config=memory_config
        )

    output_tensor = Tensor(output_tensor)
    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor


def rms_norm(input_tensor: Tensor, weight: Tensor, *, epsilon: float = 1e-6) -> Tensor:
    r"""
    rms_norm(input_tensor: Tensor) -> Tensor

    Compute rms_norm over :attr:`input_tensor`.

    """

    original_shape = input_tensor.shape
    input_tensor = _reshape_to_4D(input_tensor)
    weight = _reshape_to_4D(weight)

    ttl_input_tensor = input_tensor.value
    ttl_weight = weight.value
    ttl_output_tensor = ttl.tensor.rmsnorm(ttl_input_tensor, epsilon, ttl_weight)

    output_tensor = Tensor(ttl_output_tensor)
    output_tensor = reshape(output_tensor, original_shape)

    return output_tensor


def _torch_mean(input_tensor: Tensor, dim: int, keepdim=False, **_):
    import torch

    input_tensor = from_device(input_tensor)
    input_tensor = to_layout(input_tensor, ROW_MAJOR_LAYOUT)
    input_tensor = to_torch(input_tensor)

    return torch.mean(input_tensor, dim=dim, keepdim=keepdim)


@decorate_operation(torch_function=_torch_mean)
def mean(input_tensor: Tensor, dim: Union[int, Tuple[int]], keepdim: bool = False) -> Tensor:
    input_shape = tuple(input_tensor.shape)
    rank = len(input_shape)

    if isinstance(dim, int):
        if dim < 0:
            dim = rank + dim
        dim = (dim,)

    if isinstance(dim, tuple):
        if dim == (rank - 1,):
            reduce_op_dim = ttl.tensor.ReduceOpDim.W
        elif dim == (rank - 2,):
            reduce_op_dim = ttl.tensor.ReduceOpDim.H
        elif dim == (rank - 1, rank - 2):
            reduce_op_dim = ttl.tensor.ReduceOpDim.HW
        else:
            raise RuntimeError("Unsupported dim")
    else:
        raise RuntimeError("Invalid dim")

    output_shape = []
    padded_output_shape = []
    for axis, size in enumerate(input_shape):
        if axis in dim:
            if keepdim:
                output_shape.append(1)
                padded_output_shape.append(TILE_SIZE)
        else:
            output_shape.append(size)
            padded_output_shape.append(size)
    output_shape = tuple(output_shape)
    padded_output_shape = tuple(padded_output_shape)

    input_tensor = _reshape_to_4D(input_tensor)
    ttl_input_tensor = input_tensor.value
    ttl_output_tensor = ttl.tensor.reduce(
        ttl_input_tensor, ttl.tensor.ReduceOpMath.SUM, reduce_op_dim, 1 / input_shape[-1]
    )
    ttl_output_tensor = ttl.tensor.reduce(
        ttl_input_tensor, ttl.tensor.ReduceOpMath.SUM, reduce_op_dim, 1 / input_shape[-1]
    )

    output_tensor = Tensor(ttl_output_tensor)
    output_tensor = reshape(output_tensor, Shape(output_shape, padded_output_shape))

    return output_tensor


__all__ = [
    "matmul",
    "add",
    "sub",
    "subtract",
    "mul",
    "multiply",
    "reshape",
    "permute",
    "embedding",
    "softmax",
    "layer_norm",
    "rms_norm",
    "mean",
]
