# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pathlib
from typing import Union, Tuple, Optional, Any, Callable, Dict

from loguru import logger
import torch


import ttnn
import ttnn.decorators


def _golden_function(input_tensor: ttnn.Tensor, slices):
    output_tensor = input_tensor[slices]
    if output_tensor.ndim == 0:
        raise RuntimeError("ttnn.Tensor.__getitem__: cannot return a scalar!")
    return output_tensor


@ttnn.register_python_operation(
    name="ttnn.Tensor.__getitem__",
    is_method=True,
    golden_function=_golden_function,
)
def __getitem__(input_tensor: ttnn.Tensor, slices) -> ttnn.Tensor:
    input_rank = len(input_tensor.shape)

    if isinstance(slices, int):
        slices = (slice(None, slices, None),)
    elif isinstance(slices, slice):
        slices = (slices,)
    elif isinstance(slices, type(...)):
        return ttnn.clone(input_tensor)

    normalized_slices = []

    ellipsis_found = False
    for s in slices:
        if isinstance(s, int):
            normalized_slices.append(slice(None, s, None))
        elif isinstance(s, slice):
            normalized_slices.append(s)
        elif s is Ellipsis:
            if ellipsis_found:
                raise ValueError("Only one ellipsis ('...') is allowed in a slice.")
            ellipsis_found = True
            # Fill in the remaining dimensions with slice(None) based on how many slices are missing
            num_missing_slices = input_rank - len(slices) + 1
            normalized_slices.extend([slice(None)] * num_missing_slices)
        else:
            raise TypeError(f"Invalid slice object: {s}")

    # If fewer slices than the rank, pad with slice(None)
    while len(normalized_slices) < input_rank:
        normalized_slices.append(slice(None))

    slices = tuple(normalized_slices)

    if isinstance(slices, tuple):
        if len(slices) > input_rank:
            raise RuntimeError(f"Too many slices for tensor of rank {input_rank}")

    if input_rank <= 4:
        slice_start = [_slice.start if _slice.start is not None else 0 for _slice in slices]
        slice_end = [
            _slice.stop if _slice.stop is not None else input_tensor.shape[i] for i, _slice in enumerate(slices)
        ]
        slice_step = [_slice.step if _slice.step is not None else 1 for _slice in slices]

        output = ttnn.slice(input_tensor, slice_start, slice_end, slice_step)

        return output

    raise NotImplementedError


def _preprocess_shape(input_shape, shape):
    if isinstance(shape, tuple):
        if not (0 <= shape.count(-1) <= 1):
            raise RuntimeError("Shape cannot have more than 1 elements that is set to -1!")

        volume = math.prod(input_shape)
        new_volume = math.prod(shape)
        if new_volume < 0:
            index_of_negative_1 = shape.index(-1)
            shape = list(shape)
            shape[index_of_negative_1] = volume // (-new_volume)
            shape = tuple(shape)
        shape = ttnn.Shape(shape)
    return shape


def _preprocess_golden_function_inputs(args, kwargs):
    input_tensor, args, kwargs = ttnn.reflection.pop_argument("input_tensor", args, kwargs)
    shape, args, kwargs = ttnn.reflection.pop_argument("shape", args, kwargs)
    shape = _preprocess_shape(input_tensor.shape, shape)
    input_tensor = input_tensor.reshape(input_tensor.shape.with_tile_padding())
    return (ttnn.to_torch(input_tensor), tuple(shape.with_tile_padding()), *args), kwargs


def _golden_function(input_tensor, shape: Union[ttnn.Shape, Tuple[int, ...]]) -> ttnn.Tensor:
    return input_tensor.reshape(shape).contiguous().clone()


def _postprocess_golden_function_outputs(output, args, kwargs):
    tensor = ttnn.decorators.default_postprocess_golden_function_outputs(output, args, kwargs)

    input_tensor, args, kwargs = ttnn.reflection.pop_argument("input_tensor", args, kwargs)
    shape, args, kwargs = ttnn.reflection.pop_argument("shape", args, kwargs)
    shape = _preprocess_shape(input_tensor.shape, shape)

    shape_with_tile_padding = shape.with_tile_padding()
    if tensor.layout == ttnn.TILE_LAYOUT:
        *_, height, width = shape_with_tile_padding
        if height % ttnn.TILE_SIZE != 0 or width % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                "ttnn.reshape: cannot reshape a tensor with TILE_LAYOUT to a shape that is not a multiple of TILE_SIZE on height and width!"
            )

    tensor = ttnn.reshape(tensor, shape)

    return tensor


doc = r"""
reshape(input_tensor: ttnn.Tensor, shape: Union[Shape, Tuple[int, ...]]) -> ttnn.Tensor

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

# TODO(arakhmati): remove this once underlying C++ code can handle non-4D shapes
ttnn.register_python_operation(name="ttnn.unsqueeze_to_4D")(ttnn._ttnn.operations.core.unsqueeze_to_4D)


def _golden_function(input_tensor, *args, **kwargs):
    return input_tensor


@ttnn.register_python_operation(name="ttnn.from_torch", golden_function=_golden_function)
def from_torch(
    tensor: "torch.Tensor",
    dtype: Optional[ttnn.DataType] = None,
    *,
    tile: Optional[ttnn.Tile] = None,
    layout: Optional[ttnn.Layout] = ttnn.ROW_MAJOR_LAYOUT,
    device: Optional[ttnn.Device] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    mesh_mapper: Optional[ttnn.TensorToMesh] = None,
) -> ttnn.Tensor:
    """
    Converts the `torch.Tensor` tensor into a `ttnn.Tensor`.

    Args:
        tensor (torch.Tensor): the input tensor.
        dtype (ttnn.DataType, optional): the desired `ttnn` data type. Defaults to `None`.

    Keyword Args:
        tile (ttnn.Tile, optional): the desired tiling configuration for the tensor. Defaults to `None`.
        layout (ttnn.Layout, optional): the desired `ttnn` layout. Defaults to `ttnn.ROW_MAJOR_LAYOUT`.
        device (ttnn.Device, optional): the desired `ttnn` device. Defaults to `None`.
        memory_config (ttnn.MemoryConfig, optional): The desired `ttnn` memory configuration. Defaults to `None`.
        mesh_mapper (ttnn.TensorToMesh, optional): The desired `ttnn` mesh mapper. Defaults to `None`.

    Returns:
        ttnn.Tensor: The resulting `ttnn` tensor.

    Example:
        >>> tensor = ttnn.from_torch(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> print(tensor)
        Tensor([[1.375, -1.30469, -0.714844],
            [-0.761719, 0.53125, -0.652344]], dtype=bfloat16)
    """

    shape_with_padding = None
    if dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b:
        if len(tensor.shape) < 2:
            raise RuntimeError("ttnn.from_torch: bfloat8_b/bfloat4_b requires at least 2 dimensions!")
        if layout != ttnn.TILE_LAYOUT:
            raise RuntimeError("ttnn.from_torch: bfloat8_b/bfloat4_b requires TILE_LAYOUT!")
        # Tilize tensor
        tensor = ttnn.from_torch(tensor, layout=ttnn.TILE_LAYOUT)
        shape_with_padding = tensor.shape
        tensor = tensor.reshape(tensor.shape.with_tile_padding())
        tensor = ttnn.to_torch(tensor)

    if memory_config is not None:
        if device is None:
            raise RuntimeError("device must be specified when memory_config is specified")

    if mesh_mapper:
        shards = mesh_mapper.map(tensor)
        tensor = ttnn.Tensor(shards, dtype, mesh_mapper.config())
    else:
        if tile is not None:
            tensor = ttnn.Tensor(tensor, dtype, {}, tile)
        else:
            tensor = ttnn.Tensor(tensor, dtype)

    if layout is not None:
        tensor = ttnn.to_layout(tensor, layout, device=device)

    if device is not None:
        if memory_config is None:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
        tensor = ttnn.to_device(tensor, device, memory_config=memory_config)

    if shape_with_padding is not None and shape_with_padding != tensor.shape and mesh_mapper is None:
        tensor = ttnn.reshape(tensor, shape_with_padding)

    return tensor


def _golden_function(tensor, *, torch_rank=None, **kwargs):
    if torch_rank is None:
        return tensor

    while len(tensor.shape) != torch_rank:
        if tensor.shape[0] != 1:
            raise RuntimeError("ttnn: Unable to squeeze to desired rank!")
        tensor = tensor.squeeze()
    return tensor


class TorchTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, func_args=(), func_kwargs=None):
        # this tells torch to treat TorchTensor just like torch.Tensor's.
        # Otherwise, torch will complain that it doesn't know how to handle it.
        types = tuple(torch.Tensor if t == TorchTensor else t for t in types)
        func = ttnn._ttnn.tensor.decorate_external_operation(func, function_name=f"(torch) {func.__name__}")
        return super().__torch_function__(func, types, func_args, func_kwargs)


@ttnn.register_python_operation(name="ttnn.to_torch", golden_function=_golden_function)
def to_torch(
    tensor: ttnn.Tensor,
    *,
    torch_rank: Optional[int] = None,
    mesh_composer: Optional[ttnn.MeshToTensor] = None,
    device: Optional[ttnn.Device] = None,
    cq_id: Optional[int] = 0,
) -> "torch.Tensor":
    """
    Converts the `ttnn.Tensor` tensor into a `torch.Tensor`.

    Args:
        tensor (ttnn.Tensor): the input tensor.

    Keyword Args:
        torch_rank (int, optional): Desired rank of the `torch.Tensor`. Defaults to `None`.
            Will use `torch.squeeze` operation to remove dimensions until the desired rank is reached. If not possible, the operation will raise an error.
        mesh_composer (ttnn.MeshToTensor, optional): The desired `ttnn` mesh composer. Defaults to `None`.
        device (ttnn.Device, optional): The `ttnn` device of the input tensor. Defaults to `None`.
        cq_id (int, optional): The command queue ID to use. Defaults to `0`.

    Returns:
        torch.Tensor: The converted `torch` tensor.

    Example:
        >>> ttnn_tensor = ttnn.from_torch(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> torch_tensor = ttnn.to_torch(ttnn_tensor)
        >>> print(torch_tensor)
        tensor([[-0.3008, -0.8438,  0.3242],
                [ 0.9023, -0.5820,  0.5312]], dtype=torch.bfloat16)
    """
    if ttnn.is_tensor_storage_on_device(tensor):
        tensor = ttnn.from_device(tensor, cq_id=cq_id)

    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        tensor = tensor.to(ttnn.ROW_MAJOR_LAYOUT, device)

    shape_without_tile_padding = tuple(tensor.shape)
    if tensor.storage_type() == ttnn.DEVICE_STORAGE_TYPE:
        raise RuntimeError("ttnn.Tensor cannot be on device when converting to torch.Tensor!")
    if tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError("ttnn.Tensor has to be in ROW_MAJOR Layout to be converted to torch.Tensor")
    if mesh_composer:
        return mesh_composer.compose(tensor)
    tensor = tensor.to_torch()
    slices = [slice(None, x) for x in shape_without_tile_padding]
    tensor = tensor[slices]

    if torch_rank is not None:
        while len(tensor.shape) != torch_rank:
            if tensor.shape[0] != 1:
                raise RuntimeError("ttnn: Unable to squeeze to desired rank!")
            tensor = tensor.squeeze()

    return TorchTensor(tensor)


def _golden_function(tensor, *args, **kwargs):
    return tensor


doc = """
Copies the `ttnn.Tensor` :attr:`tensor` to the `tt_lib.device.Device`.

The tensor may be placed in DRAM or L1 memory.

Currently memory_config must be of an Interleaved tensor (not sharded)

Args:
    * :attr:`tensor`: the ttnn.Tensor
    * :attr:`device`: the ttnn.Device
    * :attr:`memory_config`: the optional MemoryConfig (DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG). Defaults to DRAM_MEMORY_CONFIG.

Example::

    >>> device_id = 0
    >>> device = ttnn.open_device(device_id=device_id)
    >>> tensor_on_host = ttnn.from_torch(torch.randn((10, 64, 32)), dtype=ttnn.bfloat16)
    >>> tensor_on_device = ttnn.to_device(tensor_on_host, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    >>> print(tensor_on_device[0,0,:3])
    Tensor([ 0.800781, -0.455078, -0.585938], dtype=bfloat16 )
"""

ttnn.register_python_operation(
    name="ttnn.to_device",
    golden_function=_golden_function,
    doc=doc,
)(ttnn._ttnn.operations.core.to_device)


def _golden_function(tensor, *args, **kwargs):
    return tensor


doc = """
Copies the `ttnn.Tensor` :attr:`tensor` to the host.

Args:
    * :attr:`tensor`: the ttnn.Tensor

Example::
    >>> device_id = 0
    >>> device = ttnn.open_device(device_id=device_id)
    >>> tensor_on_device = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
    >>> tensor_on_host = ttnn.from_device(tensor_on_device)
    >>> print(tensor_on_host[0,0,:3])
    Tensor([ 0.365234, 0.130859, 0.75], dtype=bfloat16 )
"""


ttnn.register_python_operation(
    name="ttnn.from_device",
    golden_function=_golden_function,
    doc=doc,
)(ttnn._ttnn.operations.core.from_device)

ttnn.register_python_operation(
    name="ttnn.allocate_tensor_on_device",
)(ttnn._ttnn.operations.core.allocate_tensor_on_device)

ttnn.register_python_operation(
    name="ttnn.copy_host_to_device_tensor",
)(ttnn._ttnn.operations.core.copy_host_to_device_tensor)

doc = """
Releases the resources for `ttnn.Tensor` :attr:`tensor` explicitly.

Args:
    tensor (ttnn.Tensor): The tensor whose resources will be released.
    force (bool, optional): Whether to force deallocation, even if the buffer may have multiple references. Defaults to True.

Example:
    >>> device_id = 0
    >>> device = ttnn.open_device(device_id=device_id)
    >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
    >>> tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
    >>> ttnn.deallocate(tensor)
"""

ttnn.register_python_operation(name="ttnn.deallocate", doc=doc)(ttnn._ttnn.operations.core.deallocate)


def _golden_function(tensor, *args, **kwargs):
    return tensor


ttnn.attach_golden_function(ttnn.to_memory_config, golden_function=_golden_function)


def _golden_function(tensor, *args, **kwargs):
    return tensor


ttnn.attach_golden_function(ttnn.to_layout, golden_function=_golden_function)


def _golden_function(tensor, *args, **kwargs):
    return tensor


# TODO: Merge to_dtype and typecast
ttnn.attach_golden_function(ttnn.to_dtype, golden_function=_golden_function)
ttnn.attach_golden_function(ttnn.typecast, golden_function=_golden_function)


def _golden_function(tensor, *args, **kwargs):
    return tensor


ttnn.attach_golden_function(ttnn.clone, golden_function=_golden_function)


def _golden_function(input_tensor):
    return input_tensor


ttnn.register_python_operation(name="ttnn.reallocate", golden_function=_golden_function)(
    ttnn._ttnn.operations.core.reallocate
)


@ttnn.register_python_operation(name="ttnn.load_tensor")
def load_tensor(file_name: Union[str, pathlib.Path], *, device: ttnn.Device = None) -> ttnn.Tensor:
    """
    Load tensor from a file.

    Args:
        file_name (str | pathlib.Path): the file name.

    Keyword Args:
        device (ttnn.Device, optional): the device. Defaults to `None`.

    Returns:
        ttnn.Tensor: the loaded tensor.

    Example:
        >>> device = ttnn.open_device(0)
        >>> tensor = ttnn.load_tensor(file_name=str(tensor.bin), device=device)
    """
    file_name = pathlib.Path(file_name)
    if not file_name.exists():
        raise RuntimeError(f"Unable to load the tensor from {file_name}.  The file does not exist.")
    if not file_name.is_file():
        raise RuntimeError(f"Unable to load the tensor from {file_name}.  The file is not a file.")
    return ttnn._ttnn.tensor.load_tensor(str(file_name), device)


@ttnn.register_python_operation(name="ttnn.dump_tensor")
def dump_tensor(file_name: Union[str, pathlib.Path], tensor: ttnn.Tensor, distribute: Dict[str, str] = None) -> None:
    """
    Dump tensor to a file.

    Args:
        file_name (str | pathlib.Path): The file name.
        tensor (ttnn.Tensor): the tensor to be dumped.
        distribute (Dict[str, str], optional): The distributed strategy. Only applicable to multi-device tensors. Defaults to `None`.

    Returns:
        `None`: tensor saved to a specified file.

    Example:
        >>> tensor = ttnn.ones([2, 3], bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        >>> dump_tensor(file_name=str(tensor.bin), tensor=tensor)
    """
    if distribute is None:
        distribute = dict()
    file_name = pathlib.Path(file_name)
    ttnn._ttnn.tensor.dump_tensor(str(file_name), tensor, distribute)


@ttnn.register_python_operation(name="ttnn.as_tensor")
def as_tensor(
    tensor: Union["torch.Tensor"],  # TODO: add support for numpy.ndarray and other tensor types
    dtype: Optional[ttnn.DataType] = None,
    *,
    layout: Optional[ttnn.Layout] = ttnn.ROW_MAJOR_LAYOUT,
    device: Optional[ttnn.Device] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    cache_file_name: Optional[Union[str, pathlib.Path]] = None,
    preprocess: Optional[Callable[[ttnn.Tensor], ttnn.Tensor]] = None,
    mesh_mapper: Optional[ttnn.TensorToMesh] = None,
    use_device_tilizer: bool = False,
) -> ttnn.Tensor:
    """
    Converts the `torch.Tensor` tensor into a `ttnn.Tensor`.

    Args:
        tensor (torch.Tensor): the input tensor.
        dtype (ttnn.DataType, optional): The `ttnn` data type.

    Keyword args:
        layout (ttnn.Layout, optional): The `ttnn` layout. Defaults to `ttnn.ROW_MAJOR_LAYOUT`.
        device (ttnn.Device, optional): The `ttnn` device. Defaults to `None`.
        memory_config (ttnn.MemoryConfig, optional): The `ttnn` memory configuration. Defaults to `None`.
        cache_file_name (str | pathlib.Path, optional): The cache file name. Defaults to `None`.
        preprocess (Callable[[ttnn.Tensor], ttnn.Tensor], optional): The function to preprocess the tensor before serializing/converting to ttnn. Defaults to `None`.
        mesh_mapper (ttnn.TensorToMesh, optional): The TensorToMesh to define the mapping from torch to multi-device. Defaults to `None`.
        use_device_tilizer (bool, optional): The flag that toggles whether to use host vs. device tilizer. Defaults to `False`.

            - For Grayskull, the on-device tilizer will truncate mantissa bits for bfp* formats.
            - For Wormhole, the on-device tilizer will raise a runtime error (RTE) for bfp8 but will truncate for bfp4/2 formats.

    Returns:
        ttnn.Tensor: The resulting `ttnn` tensor.

    Examples:
        >>> tensor = ttnn.as_tensor(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> print(tensor)
        Tensor([[1.375, -1.30469, -0.714844],
            [-0.761719, 0.53125, -0.652344]], dtype=bfloat16)
    """

    dtype_name = dtype.name if dtype is not None else "None"
    layout_name = layout.name if layout is not None else "None"

    if use_device_tilizer:
        if device is None:
            raise RuntimeError("device must be specified when use_device_tilizer is True")
        if memory_config is None:
            raise RuntimeError("memory_config must be specified when use_device_tilizer is True")
        if layout != ttnn.TILE_LAYOUT:
            raise RuntimeError("layout must be TILE_LAYOUT when use_device_tilizer is True")

    if device is not None and memory_config is None:
        raise RuntimeError("memory_config must be specified when device is specified")

    def torch_to_ttnn(
        tensor: torch.Tensor,
        dtype: Optional[ttnn.DataType],
        layout: Optional[ttnn.Layout],
        device: Optional[ttnn.Device],
        memory_config: Optional[ttnn.MemoryConfig],
        mesh_mapper: Optional[ttnn.TensorToMesh],
    ):
        if preprocess:
            tensor = preprocess(tensor)
        if use_device_tilizer:
            tensor = ttnn.from_torch(
                tensor,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tensor = ttnn.to_layout(tensor, layout, dtype=dtype, memory_config=memory_config, device=device)
        else:
            tensor = ttnn.from_torch(
                tensor,
                dtype=dtype,
                layout=layout,
                mesh_mapper=mesh_mapper,
                memory_config=memory_config,
                device=device,
            )
        return tensor

    if cache_file_name is None:
        return torch_to_ttnn(tensor, dtype, layout, device, memory_config, mesh_mapper)
    else:

        def from_torch_and_dump(
            tensor: torch.Tensor,
            dtype: Optional[ttnn.DataType],
            layout: Optional[ttnn.Layout],
            cache_file_name: str,
            mesh_mapper: Optional[ttnn.TensorToMesh],
        ):
            tensor = torch_to_ttnn(tensor, dtype, layout, device, memory_config, mesh_mapper)
            logger.debug(
                f"Generating cache for {cache_file_name} of shape {tensor.shape}, dtype {dtype_name}, layout {layout_name}"
            )
            pathlib.Path(cache_file_name).parent.mkdir(parents=True, exist_ok=True)
            distributed_config = mesh_mapper.config() if mesh_mapper else dict()
            ttnn._ttnn.tensor.dump_tensor(cache_file_name, tensor, distributed_config)
            return tensor

        if isinstance(mesh_mapper, ttnn.ReplicateTensorToMesh):
            storage_type = f"_multi_device" if mesh_mapper else ""
        elif mesh_mapper:
            storage_type = f"_multi_device_{device.get_num_devices()}"
        else:
            storage_type = ""

        cache_file_name = f"{cache_file_name}{storage_type}_dtype_{dtype_name}_layout_{layout_name}.bin"

        try:
            tensor = ttnn._ttnn.tensor.load_tensor(cache_file_name, device=device)
            if tuple(tensor.shape) != tuple(tensor.shape):
                logger.warning(
                    f"Cached file {cache_file_name} has shape {tensor.shape}, expected {tensor.shape}, regenerating cache"
                )
                tensor = from_torch_and_dump(tensor, dtype, layout, cache_file_name, mesh_mapper)
            logger.debug(f"Loaded cache for {cache_file_name} of shape {tensor.shape}")
        except (FileNotFoundError, RuntimeError):
            tensor = from_torch_and_dump(tensor, dtype, layout, cache_file_name, mesh_mapper)
        return tensor


__all__ = []
