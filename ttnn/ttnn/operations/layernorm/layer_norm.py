# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3 generic_op scaffold for layer_norm.

This package is intentionally isolated under ``ttnn.operations.layernorm``.
The real public ``ttnn.layer_norm`` implementation is still expected to land in
``ttnn/cpp/ttnn/operations/normalization/layernorm``.
"""

import json
from pathlib import Path

import ttnn

from .layer_norm_program_descriptor import create_program_descriptor


_ALLOWED_DTYPES = {ttnn.bfloat16, ttnn.float32}
_TDD_STATE_PATH = Path(__file__).resolve().with_name(".tdd_state.json")
_TEMP_REDUCTION_STAGES = {"mean_reduce", "invstd_reduce"}


def layer_norm(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor | None = None,
    bias: ttnn.Tensor | None = None,
    residual_input_tensor: ttnn.Tensor | None = None,
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig | None = None,
    program_config=None,
    compute_kernel_config=None,
    dtype: ttnn.DataType | None = None,
) -> ttnn.Tensor:
    stage_name = _get_current_stage_name()

    _validate_inputs(
        input_tensor,
        weight=weight,
        bias=bias,
        residual_input_tensor=residual_input_tensor,
        epsilon=epsilon,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dtype=dtype,
    )

    output_dtype = input_tensor.dtype if dtype is None else dtype
    output_memory_config = input_tensor.memory_config() if memory_config is None else memory_config

    output_shape = [int(dim) for dim in input_tensor.shape]
    if stage_name in _TEMP_REDUCTION_STAGES:
        output_shape[-1] = 32

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        output_dtype,
        input_tensor.layout,
        input_tensor.device(),
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        epsilon=epsilon,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        is_rmsnorm=False,
        stage_name=stage_name,
    )

    io_tensors = [input_tensor]
    if residual_input_tensor is not None:
        io_tensors.append(residual_input_tensor)
    if weight is not None:
        io_tensors.append(weight)
    if bias is not None:
        io_tensors.append(bias)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _get_current_stage_name() -> str:
    with _TDD_STATE_PATH.open() as state_file:
        state = json.load(state_file)

    current_stage_index = state["current_stage_index"]
    return state["stages"][current_stage_index]["name"]


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    *,
    weight: ttnn.Tensor | None,
    bias: ttnn.Tensor | None,
    residual_input_tensor: ttnn.Tensor | None,
    epsilon: float,
    memory_config: ttnn.MemoryConfig | None,
    program_config,
    compute_kernel_config,
    dtype: ttnn.DataType | None,
) -> None:
    _validate_device_tensor(input_tensor, "input_tensor")

    if len(input_tensor.shape) == 0:
        raise ValueError("layer_norm: input_tensor must have rank > 0")
    if input_tensor.padded_shape[-1] % 32 != 0 or input_tensor.padded_shape[-2] % 32 != 0:
        raise ValueError("layer_norm: input_tensor padded height and width must be tile aligned")
    if epsilon <= 0.0:
        raise ValueError("layer_norm: epsilon must be > 0")
    if dtype is not None and dtype not in _ALLOWED_DTYPES:
        raise ValueError("layer_norm: dtype must be bfloat16 or float32 when provided")
    if program_config is not None:
        raise NotImplementedError("layer_norm: non-default program_config is not supported in the Phase 3 scaffold")
    if compute_kernel_config is not None:
        raise NotImplementedError(
            "layer_norm: compute_kernel_config is reserved for the real C++ implementation and is not yet wired"
        )
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("layer_norm: output memory_config must remain INTERLEAVED in the Phase 3 scaffold")

    if residual_input_tensor is not None:
        _validate_device_tensor(residual_input_tensor, "residual_input_tensor")
        if list(residual_input_tensor.shape) != list(input_tensor.shape):
            raise ValueError("layer_norm: residual_input_tensor must match input_tensor logical shape")
        if list(residual_input_tensor.padded_shape) != list(input_tensor.padded_shape):
            raise ValueError("layer_norm: residual_input_tensor must match input_tensor padded shape")
        if residual_input_tensor.dtype != input_tensor.dtype:
            raise ValueError("layer_norm: residual_input_tensor dtype must match input_tensor dtype")

    for tensor, name in ((weight, "weight"), (bias, "bias")):
        if tensor is None:
            continue
        _validate_device_tensor(tensor, name)
        if tensor.dtype != input_tensor.dtype:
            raise ValueError(f"layer_norm: {name} dtype must match input_tensor dtype")
        if not _is_last_dim_broadcastable(tensor, input_tensor):
            raise ValueError(
                f"layer_norm: {name} must be broadcast-compatible with input_tensor and share the logical last dim"
            )
        if tensor.padded_shape[-1] != input_tensor.padded_shape[-1]:
            raise ValueError(f"layer_norm: {name} padded last dim must match input_tensor padded last dim")


def _validate_device_tensor(tensor: ttnn.Tensor, name: str) -> None:
    if tensor.storage_type() != ttnn.DEVICE_STORAGE_TYPE or not tensor.is_allocated():
        raise ValueError(f"layer_norm: {name} must be an allocated device tensor")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"layer_norm: {name} must use TILE_LAYOUT")
    if tensor.dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"layer_norm: {name} must use bfloat16 or float32")
    if tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"layer_norm: {name} must use INTERLEAVED memory")


def _is_last_dim_broadcastable(tensor: ttnn.Tensor, input_tensor: ttnn.Tensor) -> bool:
    tensor_shape = list(tensor.shape)
    input_shape = list(input_tensor.shape)
    if len(tensor_shape) != len(input_shape):
        return False
    if tensor_shape[-1] != input_shape[-1]:
        return False
    return all(tensor_dim in (1, input_dim) for tensor_dim, input_dim in zip(tensor_shape[:-1], input_shape[:-1]))
