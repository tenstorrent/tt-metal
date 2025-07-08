# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ttnn.decorators
from loguru import logger

import ttnn


def _golden_function(input_tensor: ttnn.Tensor, slices):
    output_tensor = input_tensor[slices]
    if output_tensor.ndim == 0:
        raise RuntimeError("ttnn.Tensor.__getitem__: cannot return a scalar!")
    return output_tensor


def _host_slice_with_unpad(input_tensor: ttnn.Tensor, begins, ends) -> ttnn.Tensor:
    """Hacky fallback to old `unpad` methods for host based accessing"""

    working_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT).unpad(begins, ends)
    working_tensor = ttnn.view(working_tensor, [e - b for e, b in zip(ends, begins)])
    return ttnn.to_layout(working_tensor, input_tensor.get_layout())


@ttnn.register_python_operation(
    name="ttnn.Tensor.__getitem__",
    is_method=True,
    golden_function=_golden_function,
)
def __getitem__(input_tensor: ttnn.Tensor, slices) -> ttnn.Tensor:
    """
    Mimics PyTorch-style indexing for ttnn.Tensor using ttnn.slice and ttnn.squeeze.
    """

    # Gather some basic info
    input_rank = len(input_tensor.shape)
    shape = input_tensor.shape  # Or input_tensor.logical_shape, depending on your library's conventions

    # 1) Normalize slices into a tuple.
    #    e.g. if user wrote a[3], slices is just int(3), so wrap it into (3,).
    #    or if user wrote a[2:5], slices is slice(2,5).
    if isinstance(slices, (int, slice, type(...))):
        slices = (slices,)
    else:
        # ensure it's a tuple in case user wrote something like a[1, 2:5, ...]
        slices = tuple(slices)

    # 2) Expand any bare Ellipsis into enough slice(None) to fill out to input_rank.
    #    But we have to do this carefully in the presence of other slices.
    #    We'll do it in two passes:
    #      - first copy slices to a “normalized_slices”, remembering where Ellipsis is
    #      - then replace the Ellipsis with however many slice(None) are needed
    normalized_slices = []
    ellipsis_found = False

    for s in slices:
        if s is Ellipsis:
            if ellipsis_found:
                raise ValueError("Only one ellipsis ('...') is allowed in a slice.")
            # We'll deal with actually expanding it after this loop.
            ellipsis_found = True
            normalized_slices.append(Ellipsis)
        else:
            normalized_slices.append(s)

    # If there's exactly one Ellipsis, expand it
    if ellipsis_found:
        ellipsis_index = normalized_slices.index(Ellipsis)
        # Number of slices ignoring the Ellipsis
        num_slices_no_ellipsis = len(normalized_slices) - 1
        # How many dimensions are “missing”
        num_missing = input_rank - num_slices_no_ellipsis
        if num_missing < 0:
            raise IndexError(f"Too many indices for tensor of dimension {input_rank}")

        # Remove the Ellipsis placeholder
        del normalized_slices[ellipsis_index]
        # Insert slice(None) for however many dims are missing
        for _ in range(num_missing):
            normalized_slices.insert(ellipsis_index, slice(None, None, None))

    # If there was no Ellipsis and we still have fewer slices than rank, pad with slice(None)
    while len(normalized_slices) < input_rank:
        normalized_slices.append(slice(None, None, None))

    # Now if we have more slices than the rank, that’s an error
    if len(normalized_slices) > input_rank:
        raise IndexError(f"Too many indices for tensor of dimension {input_rank}")

    # 3) Convert everything into slice objects (including integer indices),
    #    and record which dimensions we’ll need to squeeze out (integer-indexed dims).
    final_slices = []
    singled_out_dims = []  # dims where user gave an integer index

    for dim_idx, s in enumerate(normalized_slices):
        if isinstance(s, int):
            # Negative index => convert as in python: s + size if s < 0
            idx = s if s >= 0 else (s + shape[dim_idx])
            if not 0 <= idx < shape[dim_idx]:
                raise IndexError(
                    f"Index {s} (converted to {idx}) is out of bounds "
                    f"for dimension {dim_idx} of size {shape[dim_idx]}"
                )
            final_slices.append(slice(idx, idx + 1, 1))
            singled_out_dims.append(dim_idx)
        elif isinstance(s, slice):
            # We mimic Python negative slicing for start/stop
            start, stop, step = s.start, s.stop, s.step

            # default values
            if start is None:
                start = 0
            if stop is None:
                stop = shape[dim_idx]
            if step is None:
                step = 1

            final_slices.append(slice(start, stop, step))
        else:
            raise TypeError(f"Invalid slice type: {s}")

    # 4) Prepare the lists for ttnn.slice
    slice_start = []
    slice_end = []
    slice_step = []

    for dim_idx, sl in enumerate(final_slices):
        # No further negative indexing needed: we already converted above
        slice_start.append(sl.start)
        slice_end.append(sl.stop)
        slice_step.append(sl.step)

    # 5) Perform the slicing
    if ttnn.is_tensor_storage_on_device(input_tensor):
        output = ttnn.slice(input_tensor, slice_start, slice_end, slice_step)
    else:
        if not all([s == 1 for s in slice_step]):
            raise RuntimeError("Host tensors cannot be accessed with non-unit stride")
        output = _host_slice_with_unpad(input_tensor, slice_start, slice_end)

    # 6) Squeeze out all dimensions that were indexed by an integer.
    #    We do this from left to right, adjusting each subsequent dimension index
    #    or do it from right to left so we don't need to adjust. Either approach works
    #    if we’re careful. The simplest is ascending order with a small counter.
    shift = 0
    for original_dim in sorted(singled_out_dims):
        # after removing 'shift' dims, the current dim is (original_dim - shift)
        squeeze_dim = original_dim - shift
        # Squeeze only if that dim is actually size 1 after slicing
        output = ttnn.squeeze(output, squeeze_dim)
        shift += 1

    return output


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
    input_tensor = input_tensor.reshape(input_tensor.padded_shape)
    return (ttnn.to_torch(input_tensor), tuple(shape), *args), kwargs


def _golden_function(input_tensor, shape: Union[ttnn.Shape, Tuple[int, ...]]) -> ttnn.Tensor:
    return input_tensor.reshape(shape).contiguous().clone()


def _postprocess_golden_function_outputs(output, args, kwargs):
    tensor = ttnn.decorators.default_postprocess_golden_function_outputs(output, args, kwargs)

    input_tensor, args, kwargs = ttnn.reflection.pop_argument("input_tensor", args, kwargs)
    shape, args, kwargs = ttnn.reflection.pop_argument("shape", args, kwargs)
    shape = _preprocess_shape(input_tensor.shape, shape)

    if tensor.layout == ttnn.TILE_LAYOUT:
        *_, height, width = shape
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
ttnn.attach_golden_function(ttnn.reshape, golden_function=_golden_function)

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
    pad_value: Optional[float] = None,
    layout: Optional[ttnn.Layout] = ttnn.ROW_MAJOR_LAYOUT,
    device: Optional[ttnn.MeshDevice] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    mesh_mapper: Optional[ttnn.CppTensorToMesh | ttnn.ReplicateTensorToMeshWrapper] = None,
    cq_id: Optional[int] = ttnn.DefaultQueueId,
) -> ttnn.Tensor:
    """
    Converts the `torch.Tensor` tensor into a `ttnn.Tensor`. For bfloat8_b or bfloat4_b format, the function itself is called twice,
    first call runs in bfloat16 format, and calls to_layout to convert from row_major layout to tile layout (for padding purpose in case input
    is not tile padded). Second call runs in desired format and does not call to_layout for bfloat8_b or bfloat4_b as we now convert
    to tile layout during tensor creation (ttnn.Tensor).

    Args:
        tensor (torch.Tensor): the input tensor.
        dtype (ttnn.DataType, optional): the desired `ttnn` data type. Defaults to `None`.

    Keyword Args:
        tile (ttnn.Tile, optional): the desired tiling configuration for the tensor. Defaults to `None`.
        pad_value (float, optional): the desired padding value for tiling. Only used if `layout` is `TILE_LAYOUT`. Defaults to `None`.
        layout (ttnn.Layout, optional): the desired `ttnn` layout. Defaults to `ttnn.ROW_MAJOR_LAYOUT`.
        device (ttnn.MeshDevice, optional): the desired `ttnn` device. Defaults to `None`.
        memory_config (ttnn.MemoryConfig, optional): The desired `ttnn` memory configuration. Defaults to `None`.
        mesh_mapper (ttnn.TensorToMesh, optional): The desired `ttnn` mesh mapper. Defaults to `None`.
        cq_id (int, optional): The command queue ID to use. Defaults to `0`.

    Returns:
        ttnn.Tensor: The resulting `ttnn` tensor.

    Example:
        >>> tensor = ttnn.from_torch(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> print(tensor)
        Tensor([[1.375, -1.30469, -0.714844],
            [-0.761719, 0.53125, -0.652344]], dtype=bfloat16)
    """
    if memory_config is not None and memory_config.is_sharded():
        if memory_config.shard_spec is None and memory_config.nd_shard_spec is None:
            raise RuntimeError("ttnn.from_torch: Shard spec must not be None for sharded tensors")

    if dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b:
        if layout != ttnn.TILE_LAYOUT:
            raise RuntimeError("ttnn.from_torch: bfloat8_b/bfloat4_b requires TILE_LAYOUT!")

    if memory_config is not None and device is None:
        raise RuntimeError("ttnn.from_torch: device must be specified when memory_config is specified")

    return ttnn.Tensor(
        tensor=tensor,
        data_type=dtype,
        device=device,
        layout=layout,
        mem_config=memory_config,
        tile=tile,
        cq_id=cq_id,
        pad_value=pad_value,
        mesh_mapper=mesh_mapper.unwrap() if isinstance(mesh_mapper, ttnn.ReplicateTensorToMeshWrapper) else mesh_mapper,
    )


def _golden_function(tensor, *, torch_rank=None, **kwargs):
    if torch_rank is None:
        return tensor

    while len(tensor.shape) > torch_rank:
        if tensor.shape[0] != 1:
            raise RuntimeError("ttnn: Unable to squeeze to desired rank!")
        tensor = tensor.squeeze(0)
    return tensor


@ttnn.register_python_operation(name="ttnn.to_torch", golden_function=_golden_function)
def to_torch(
    tensor: ttnn.Tensor,
    dtype: Optional["torch.dtype"] = None,
    *,
    torch_rank: Optional[int] = None,
    mesh_composer: Optional[ttnn.CppMeshToTensor] = None,
    device: Optional[ttnn.MeshDevice] = None,
    cq_id: Optional[int] = ttnn.DefaultQueueId,
) -> "torch.Tensor":
    """
    Converts the `ttnn.Tensor` tensor into a `torch.Tensor`. It does not call to_layout for bfloat8_b or bfloat4_b as we now convert
    to tile layout during tensor.to_torch().

    Args:
        tensor (ttnn.Tensor): the input tensor.
        dtype (torch.dtype, optional): the desired `torch` data type of returned tensor. Defaults to `None`.

    Keyword Args:
        torch_rank (int, optional): Desired rank of the `torch.Tensor`. Defaults to `None`.
            Will use `torch.squeeze` operation to remove dimensions until the desired rank is reached. If not possible, the operation will raise an error.
        mesh_composer (ttnn.CppMeshToTensor, optional): The desired `ttnn` mesh composer. Defaults to `None`.
        device (ttnn.MeshDevice, optional): The `ttnn` device of the input tensor. Defaults to `None`.
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
    import torch

    if ttnn.is_tensor_storage_on_device(tensor):
        tensor = ttnn.from_device(tensor, cq_id=cq_id)

    tensor = tensor.to_torch(mesh_composer=mesh_composer)

    if torch_rank is not None:
        while len(tensor.shape) > torch_rank:
            if tensor.shape[0] != 1:
                raise RuntimeError("ttnn: Unable to squeeze to desired rank!")
            tensor = tensor.squeeze(0)

    torch_tensor = tensor

    if dtype is not None:
        torch_tensor = torch_tensor.to(dtype=dtype)

    return torch_tensor


def _golden_function(tensor, *args, **kwargs):
    return tensor


doc = """
Copies the `ttnn.Tensor` :attr:`tensor` to the `tt_lib.device.MeshDevice`.

The tensor may be placed in DRAM or L1 memory.

Currently memory_config must be of an Interleaved tensor (not sharded)

Args:
    * :attr:`tensor`: the ttnn.Tensor
    * :attr:`device`: the ttnn.MeshDevice
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
    name="ttnn.allocate_tensor_on_host",
)(ttnn._ttnn.operations.core.allocate_tensor_on_host)

ttnn.register_python_operation(
    name="ttnn.copy_host_to_device_tensor",
)(ttnn._ttnn.operations.core.copy_host_to_device_tensor)
ttnn.register_python_operation(
    name="ttnn.copy_device_to_host_tensor",
)(ttnn._ttnn.operations.core.copy_device_to_host_tensor)

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


def _golden_function(input_tensor, *args, **kwargs):
    return input_tensor


ttnn.register_python_operation(name="ttnn.reallocate", golden_function=_golden_function)(
    ttnn._ttnn.operations.core.reallocate
)

ttnn.attach_golden_function(ttnn.reallocate, golden_function=_golden_function)


@ttnn.register_python_operation(name="ttnn.load_tensor")
def load_tensor(file_name: Union[str, pathlib.Path], *, device: ttnn.MeshDevice = None) -> ttnn.Tensor:
    """
    Load tensor from a file.

    Args:
        file_name (str | pathlib.Path): the file name.

    Keyword Args:
        device (ttnn.MeshDevice, optional): the device. Defaults to `None`.

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
def dump_tensor(file_name: Union[str, pathlib.Path], tensor: ttnn.Tensor) -> None:
    """
    Dump tensor to a file.

    Args:
        file_name (str | pathlib.Path): The file name.
        tensor (ttnn.Tensor): the tensor to be dumped.

    Returns:
        `None`: tensor saved to a specified file.

    Example:
        >>> tensor = ttnn.ones([2, 3], bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        >>> dump_tensor(file_name=str(tensor.bin), tensor=tensor)
    """
    file_name = pathlib.Path(file_name)
    ttnn._ttnn.tensor.dump_tensor(str(file_name), tensor)


@ttnn.register_python_operation(name="ttnn.as_tensor")
def as_tensor(
    tensor: Union["torch.Tensor"],  # TODO: add support for numpy.ndarray and other tensor types
    dtype: Optional[ttnn.DataType] = None,
    *,
    layout: Optional[ttnn.Layout] = ttnn.ROW_MAJOR_LAYOUT,
    device: Optional[ttnn.MeshDevice] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    cache_file_name: Optional[Union[str, pathlib.Path]] = None,
    preprocess: Optional[Callable[[ttnn.Tensor], ttnn.Tensor]] = None,
    mesh_mapper: Optional[ttnn.CppTensorToMesh | ttnn.ReplicateTensorToMeshWrapper] = None,
    use_device_tilizer: bool = False,
) -> ttnn.Tensor:
    """
    Converts the `torch.Tensor` tensor into a `ttnn.Tensor`.

    Args:
        tensor (torch.Tensor): the input tensor.
        dtype (ttnn.DataType, optional): The `ttnn` data type.

    Keyword args:
        layout (ttnn.Layout, optional): The `ttnn` layout. Defaults to `ttnn.ROW_MAJOR_LAYOUT`.
        device (ttnn.MeshDevice, optional): The `ttnn` device. Defaults to `None`.
        memory_config (ttnn.MemoryConfig, optional): The `ttnn` memory configuration. Defaults to `None`.
        cache_file_name (str | pathlib.Path, optional): The cache file name. Defaults to `None`.
        preprocess (Callable[[ttnn.Tensor], ttnn.Tensor], optional): The function to preprocess the tensor before serializing/converting to ttnn. Defaults to `None`.
        mesh_mapper (ttnn.CppTensorToMesh, optional): The TensorToMesh to define the mapping from torch to multi-device. Defaults to `None`.
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
        tensor: "torch.Tensor",
        dtype: Optional[ttnn.DataType],
        layout: Optional[ttnn.Layout],
        device: Optional[ttnn.MeshDevice],
        memory_config: Optional[ttnn.MemoryConfig],
        mesh_mapper: Optional[ttnn.CppTensorToMesh | ttnn.ReplicateTensorToMeshWrapper],
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
            tensor = ttnn.to_layout(tensor, layout, dtype=dtype, memory_config=memory_config)
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
            tensor: "torch.Tensor",
            dtype: Optional[ttnn.DataType],
            layout: Optional[ttnn.Layout],
            cache_file_name: str,
            mesh_mapper: Optional[ttnn.CppTensorToMesh | ttnn.ReplicateTensorToMeshWrapper],
        ):
            tensor = torch_to_ttnn(tensor, dtype, layout, device, memory_config, mesh_mapper)
            logger.debug(
                f"Generating cache for {cache_file_name} of shape {tensor.shape}, dtype {dtype_name}, layout {layout_name}"
            )
            pathlib.Path(cache_file_name).parent.mkdir(parents=True, exist_ok=True)
            ttnn._ttnn.tensor.dump_tensor(cache_file_name, tensor)
            return tensor

        if isinstance(mesh_mapper, ttnn.ReplicateTensorToMeshWrapper):
            storage_type = f"_multi_device" if mesh_mapper else ""
        elif mesh_mapper:
            storage_type = f"_multi_device_{device.get_num_devices()}"
        else:
            storage_type = ""

        cache_file_name = f"{cache_file_name}{storage_type}_dtype_{dtype_name}_layout_{layout_name}.bin"

        cache_path = pathlib.Path(cache_file_name)

        # Prepend "tt-mesh" to differentiate file store for new weights format
        cache_path = cache_path.parent / "tt-mesh" / cache_path.name
        cache_file_name = str(cache_path)

        if not cache_path.exists() or not cache_path.is_file():
            return from_torch_and_dump(tensor, dtype, layout, cache_file_name, mesh_mapper)

        try:
            tensor = ttnn._ttnn.tensor.load_tensor(cache_file_name, device=device)
            if tuple(tensor.shape) != tuple(tensor.shape):
                logger.warning(
                    f"Cached file {cache_file_name} has shape {tensor.shape}, expected {tensor.shape}, regenerating cache"
                )
                tensor = from_torch_and_dump(tensor, dtype, layout, cache_file_name, mesh_mapper)
            logger.debug(f"Loaded cache for {cache_file_name} of shape {tensor.shape}")
        except RuntimeError as e:
            logger.warning(f"Failed to load cache for {cache_file_name}: {e}")
            tensor = from_torch_and_dump(tensor, dtype, layout, cache_file_name, mesh_mapper)
        return tensor


__all__ = []
