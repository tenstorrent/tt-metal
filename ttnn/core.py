# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pathlib
from typing import Optional, Tuple, Union

from loguru import logger

import tt_lib as ttl

from ttnn.tensor import (
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
)

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
    if len(tensor.shape) > 4:
        raise RuntimeError("Tensor cannot have more than 4 dimensions!")
    num_missing_dims = 4 - len(tensor.shape)
    shape = tuple(([1] * num_missing_dims) + tensor.shape)
    return reshape(tensor, shape=shape)


# Math Operations


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
    output_shape = tuple(input_shape_a[:-1] + input_shape_b[-1:])

    if not isinstance(input_tensor_a, Tensor):
        raise RuntimeError("Expected first argument to be a ttnn.Tensor")
    if not isinstance(input_tensor_b, Tensor):
        raise RuntimeError("Expected second argument to be a ttnn.Tensor")

    if input_tensor_a._tensor.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if input_tensor_b._tensor.storage_type() != ttl.tensor.StorageType.DEVICE:
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

    input_tensor_a = reshape(input_tensor_a, tuple(batch_shape_a + [height_a, width_a]))
    input_tensor_b = reshape(input_tensor_b, tuple(batch_shape_b + [height_b, width_b]))

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

        ttl_input_tensor_a = input_tensor_a._tensor
        ttl_input_tensor_b = input_tensor_b._tensor
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
        input_tensor_b = reshape(input_tensor_b, tuple(input_tensor_b.shape[:-2] + [width_b, height_b]))

        ttl_input_tensor_a = input_tensor_a._tensor
        ttl_input_tensor_b = input_tensor_b._tensor

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
            ttl_input_tensor_a = input_tensor_a._tensor
            ttl_input_tensor_b = input_tensor_b._tensor
            output_tensor = Tensor(
                ttl.tensor.matmul(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)
            )
        else:
            ttl_input_tensor_a = input_tensor_a._tensor
            ttl_input_tensor_b = input_tensor_b._tensor
            output_tensor = Tensor(
                ttl.tensor.bmm(ttl_input_tensor_a, ttl_input_tensor_b, output_mem_config=memory_config)
            )

    else:
        raise RuntimeError("These tensors cannot be broadcasted")

    if output_tensor.shape != output_shape:
        output_tensor = reshape(output_tensor, output_shape)
    return output_tensor


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
        >>> weights = ttnn.to_device(ttnn.from_torch(torch.randn((32, 128), dtype=torch.bfloat16)), device)
        >>> bias = ttnn.to_device(ttnn.from_torch(torch.randn((128,), dtype=torch.bfloat16)), device)
        >>> output = torch.linear(activations, weights, bias=bias)
        >>> print(output.shape)
        [10, 64, 128]
    """

    if dtype is None:
        dtype = input_tensor_a.dtype

    input_shape_a = input_tensor_a.shape
    input_shape_b = input_tensor_b.shape
    output_shape = tuple(input_shape_a[:-1] + input_shape_b[-1:])

    if not isinstance(input_tensor_a, Tensor):
        raise RuntimeError("Expected first argument to be a ttnn.Tensor")
    if not isinstance(input_tensor_b, Tensor):
        raise RuntimeError("Expected second argument to be a ttnn.Tensor")

    if input_tensor_a._tensor.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if input_tensor_b._tensor.storage_type() != ttl.tensor.StorageType.DEVICE:
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

    input_tensor_a = reshape(input_tensor_a, tuple(batch_shape_a + [height_a, width_a]))
    input_tensor_b = reshape(input_tensor_b, tuple(batch_shape_b + [height_b, width_b]))

    input_tensor_a = _reshape_to_4D(input_tensor_a)
    input_tensor_b = _reshape_to_4D(input_tensor_b)

    if width_a != height_b:
        raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")

    m_size = height_a
    k_size = width_a
    n_size = width_b

    ttl_input_tensor_a = input_tensor_a._tensor
    ttl_input_tensor_b = input_tensor_b._tensor

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
            ttl_bias = bias._tensor if bias is not None else None
            if activation == "gelu":
                fused_activation = (ttl.tensor.FusibleActivation.GELU, True)
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
            raise RuntimeError("activations must be None")

        ttl_bias = bias._tensor if bias is not None else None
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


def add(input_tensor_a: Tensor, input_tensor_b: Union[Tensor, int, float], *, alpha: Union[int, float] = 1) -> Tensor:
    """
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

    original_shape = tuple(input_tensor_a.shape)
    input_tensor_a = _reshape_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a._tensor

    if not input_tensor_a.is_on_device:
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        output_tensor = Tensor(ttl.tensor.add_unary(ttl_input_tensor_a, input_tensor_b * alpha))
        return reshape(output_tensor, original_shape)
    elif isinstance(input_tensor_b, Tensor):
        input_shape_b = input_tensor_b._tensor.shape_without_padding()

        if len(input_shape_b) == 1:
            height_b = 1
            (width_b,) = input_shape_b
        else:
            *_, height_b, width_b = input_shape_b

        input_tensor_b = _reshape_to_4D(input_tensor_b)
        ttl_input_tensor_b = input_tensor_b._tensor
        if ttl_input_tensor_b.storage_type() != ttl.tensor.StorageType.DEVICE:
            raise RuntimeError("input_tensor_a must be on device!")
    else:
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    ttl_input_tensor_b = input_tensor_b._tensor
    if alpha != 1:
        ttl_input_tensor_b = ttl.tensor.mul_unary(ttl_input_tensor_b, alpha)

    if height_b == 1 and width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW
            )
        )
    elif height_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H
            )
        )
    elif width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.W
            )
        )
    else:
        output_tensor = Tensor(ttl.tensor.add(ttl_input_tensor_a, ttl_input_tensor_b))

    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor


def sub(input_tensor_a: Tensor, input_tensor_b: Union[Tensor, int, float], *, alpha: Union[int, float] = 1) -> Tensor:
    """
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
    ttl_input_tensor_a = input_tensor_a._tensor

    if ttl_input_tensor_a.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        output_tensor = Tensor(ttl.tensor.sub_unary(ttl_input_tensor_a, input_tensor_b * alpha))
        return reshape(output_tensor, original_shape)
    elif isinstance(input_tensor_b, Tensor):
        input_tensor_b = _reshape_to_4D(input_tensor_b)
        ttl_input_tensor_b = input_tensor_b._tensor
        if ttl_input_tensor_b.storage_type() != ttl.tensor.StorageType.DEVICE:
            raise RuntimeError("input_tensor_a must be on device!")
    else:
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    ttl_input_tensor_b = input_tensor_b._tensor
    input_shape_b = ttl_input_tensor_b.shape()

    if alpha != 1:
        ttl_input_tensor_b = ttl.tensor.mul_unary(ttl_input_tensor_b, alpha)

    if len(input_shape_b) == 1:
        height_b = 1
        (width_b,) = input_shape_b
    else:
        *_, height_b, width_b = input_shape_b

    if height_b == 1 and width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.HW
            )
        )
    elif height_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.H
            )
        )
    elif width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(
                ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.W
            )
        )
    else:
        output_tensor = Tensor(ttl.tensor.sub(ttl_input_tensor_a, ttl_input_tensor_b))

    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor


def mul(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor:
    """
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

    original_shape = tuple(input_tensor_a.shape)
    input_tensor_a = _reshape_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a._tensor

    if not isinstance(input_tensor_a, Tensor):
        raise TypeError("Expected first argument to be a ttnn.Tensor")

    ttl_input_tensor_a = input_tensor_a._tensor

    if not input_tensor_a.is_on_device:
        raise RuntimeError("input_tensor_a must be on device!")

    if _is_scalar(input_tensor_b):
        return reshape(Tensor(ttl.tensor.mul_unary(ttl_input_tensor_a, input_tensor_b)), original_shape)
    elif not isinstance(input_tensor_b, Tensor):
        raise TypeError("Expected second argument to be a ttnn.Tensor or a scalar")

    input_tensor_b = _reshape_to_4D(input_tensor_b)
    ttl_input_tensor_b = input_tensor_b._tensor
    input_shape_b = ttl_input_tensor_b.shape()
    *_, height_b, width_b = input_shape_b

    if height_b == 1 and width_b == 1:
        return reshape(
            Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW
                ),
                original_shape,
            )
        )
    elif height_b == 1:
        return reshape(
            Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H
                )
            ),
            original_shape,
        )
    elif width_b == 1:
        return reshape(
            Tensor(
                ttl.tensor.bcast(
                    ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W
                )
            ),
            original_shape,
        )

    return reshape(Tensor(ttl.tensor.mul(ttl_input_tensor_a, ttl_input_tensor_b)), original_shape)


def tanh(input_tensor: Tensor) -> Tensor:
    """
    mul(input_tensor: Tensor) -> Tensor

    Applies tanh to :attr:`input_tensor` element-wise.

    .. math::
        tanh(\mathrm{{input\_tensor}}_i)

    Args:
        * :attr:`input_tensor`

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
        >>> output = ttnn.tanh(tensor)
        >>> print(output)
        Tensor([ 0, 2], dtype=bfloat16 )

    """

    original_shape = tuple(input_tensor.shape)
    input_tensor = _reshape_to_4D(input_tensor)
    ttl_input_tensor = input_tensor._tensor

    if not isinstance(input_tensor, Tensor):
        raise TypeError("Expected first argument to be a ttnn.Tensor")

    ttl_input_tensor = input_tensor._tensor

    if not input_tensor.is_on_device:
        raise RuntimeError("input_tensor must be on device!")

    return reshape(Tensor(ttl.tensor.tanh(ttl_input_tensor)), original_shape)


subtract = sub
multiply = mul


Tensor.__matmul__ = matmul
Tensor.__add__ = add
Tensor.__radd__ = add
Tensor.__sub__ = sub
Tensor.__mul__ = mul
Tensor.__rmul__ = mul


# Data Transformations
def reshape(input_tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    reshape(input_tensor: Tensor, shape: Tuple[int, ...]) -> Tensor

    Reshape :attr:`input_tensor` into :attr:`shape`.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`shape`: the desired shape.

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.reshape(tensor, (1, 1, 32, 64))
        >>> print(output.shape)
        [1, 1, 32, 64]

    """

    if not isinstance(shape, tuple):
        raise RuntimeError("order must be a tuple")

    ttl_input_tensor = input_tensor._tensor

    if input_tensor.shape == shape:
        return input_tensor

    def ttnn_reshape(ttl_input_tensor, shape):
        return Tensor(ttl_input_tensor.reshape(shape))

    if input_tensor.layout == ROW_MAJOR_LAYOUT:
        # TODO(arakhmati): figure out how to make this work
        if input_tensor.shape != [1, 64, 4, 32] and input_tensor.shape != [8, 384, 16, 64]:
            return ttl.tensor.decorate_external_operation(ttnn_reshape, function_name="ttnn.reshape")(
                ttl_input_tensor, shape
            )

    if input_tensor.layout == TILE_LAYOUT:
        *_, old_height, old_width = input_tensor.shape
        *_, new_height, new_width = shape
        if (
            old_height % TILE_SIZE == 0
            and old_width % TILE_SIZE == 0
            and new_height % TILE_SIZE == 0
            and new_width % TILE_SIZE == 0
        ):
            return ttl.tensor.decorate_external_operation(ttnn_reshape, function_name="ttnn.reshape")(
                ttl_input_tensor, shape
            )

    try:
        w, z, y, x = shape
        return Tensor(ttl.tensor.reshape(ttl_input_tensor, w, z, y, x))
    except:

        def torch_reshape(tensor, shape):
            return tensor.reshape(shape).contiguous()

        device = ttl_input_tensor.device()
        tensor = to_layout(input_tensor, ROW_MAJOR_LAYOUT)
        tensor = from_device(tensor)
        tensor = to_torch(tensor)
        tensor = torch_reshape(tensor, shape)
        tensor = ttl.tensor.decorate_external_operation(torch_reshape, function_name="torch.reshape")(tensor, shape)
        tensor = from_torch(tensor, input_tensor.dtype)
        tensor = to_device(tensor, device)
        return tensor


def permute(input_tensor: Tensor, order: Tuple[int, ...]) -> Tensor:
    """
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

    if not input_tensor.is_on_device:
        RuntimeError("input_tensor must be on device!")

    ttl_input_tensor = input_tensor._tensor

    try:
        return Tensor(ttl.tensor.permute(ttl_input_tensor, order))
    except:

        def torch_permute(tensor, order):
            return tensor.permute(order).contiguous()

        device = ttl_input_tensor.device()
        tensor = to_layout(input_tensor, ROW_MAJOR_LAYOUT)
        tensor = from_device(tensor)
        tensor = to_torch(tensor)
        tensor = ttl.tensor.decorate_external_operation(torch_permute, function_name="torch.permute")(tensor, order)
        tensor = from_torch(tensor, input_tensor.dtype)
        tensor = to_device(tensor, device)
        return tensor


def softmax(input_tensor: Tensor, dim: int) -> Tensor:
    """
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

    rank = len(input_tensor.shape)
    if dim < 0:
        dim = rank + dim
    if dim != rank - 1:
        raise RuntimeError("Softmax can only operate on the last dimension.")

    ttl_input_tensor = input_tensor._tensor
    ttl_output_tensor = ttl.tensor.softmax(ttl_input_tensor)
    return Tensor(ttl_output_tensor)


def embedding(
    input_tensor: Tensor,
    weights: Tensor,
    *,
    layout: Layout = ROW_MAJOR_LAYOUT,
    memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
):
    """
    embedding(input_tensor: ttnn.Tensor, weights: ttnn.Tensor) -> None

    Retrieves word embeddings using input_tensor. The input_tensor is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.

    Args:
        * :attr:`input_tensor`: the indices ttnn.Tensor
        * :attr:`weights`: the embeddings ttnn.Tensor that correspond to the indices ttnn.Tensor

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> input_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device)
        >>> # an embedding matrix containing 10 tensors of size 4
        >>> weights = ttnn.to_device(ttnn.from_torch(torch.rand(10, 4), dtype=ttnn.bfloat16), device)
        >>> ttnn.embedding(input_tensor, weights)
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
        raise RuntimeError("Input Tensor must have strictly 2 dimensions!")
    if len(weights.shape) != 2:
        raise RuntimeError("Weight Tensor must have strictly 2 dimensions!")

    *_, hidden_embedding_dim = tuple(weights.shape)
    weights = _reshape_to_4D(weights)

    *_, batch_size, sentence_size = input_tensor.shape
    input_tensor = reshape(input_tensor, shape=(batch_size, 1, sentence_size, 1))

    split_weights = False
    tilized = layout == TILE_LAYOUT
    embeddings = Tensor(
        ttl.tensor.embeddings(input_tensor._tensor, weights._tensor, split_weights, tilized, memory_config)
    )
    embeddings = reshape(embeddings, shape=(batch_size, sentence_size, hidden_embedding_dim))

    return embeddings


__all__ = ["matmul", "add", "sub", "subtract", "mul", "multiply", "reshape", "permute", "softmax", "embedding", "tanh"]
