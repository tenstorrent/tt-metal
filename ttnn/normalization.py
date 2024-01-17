# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import ttnn.tensor as ttnn

import tt_lib as ttl


def _torch_layer_norm(
    input_tensor: ttnn.Tensor, *, epsilon=1e-12, residual_input_tensor=None, weight=None, bias=None, **_
):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    if residual_input_tensor is not None:
        residual_input_tensor = ttnn.from_device(residual_input_tensor)
        residual_input_tensor = ttnn.to_layout(residual_input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        residual_input_tensor = ttnn.to_torch(residual_input_tensor)
        input_tensor += residual_input_tensor

    if weight is not None:
        weight = ttnn.from_device(weight)
        weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        weight = ttnn.to_torch(weight)
        if len(weight.shape) == 2:
            weight = weight[0]

    if bias is not None:
        bias = ttnn.from_device(bias)
        bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.to_torch(bias)
        if len(bias.shape) == 2:
            bias = bias[0]

    return torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight, bias, eps=epsilon)


@ttnn.decorate_operation(torch_function=_torch_layer_norm)
def layer_norm(
    input_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-12,
    residual_input_tensor: Optional[ttnn.Tensor] = None,
    weight: Optional[ttnn.Tensor] = None,
    bias: Optional[ttnn.Tensor] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    layer_norm(input_tensor: ttnn.Tensor) -> ttnn.Tensor

    Compute layer_norm over :attr:`input_tensor`.

    """

    if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
        raise RuntimeError("layer_norm only supports device storage type")

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    if residual_input_tensor is not None:
        residual_input_tensor = ttnn.unsqueeze_to_4D(residual_input_tensor)
    if weight is not None:
        weight = ttnn.unsqueeze_to_4D(weight)
    if bias is not None:
        bias = ttnn.unsqueeze_to_4D(bias)

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

    output_tensor = ttnn.Tensor(output_tensor)
    output_tensor = ttnn.reshape(output_tensor, original_shape)
    return output_tensor


def rms_norm(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, epsilon: float = 1e-6) -> ttnn.Tensor:
    r"""
    rms_norm(input_tensor: ttnn.Tensor) -> ttnn.Tensor

    Compute rms_norm over :attr:`input_tensor`.

    """

    if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
        raise RuntimeError("rms_norm only supports device storage type")

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    weight = ttnn.unsqueeze_to_4D(weight)

    ttl_input_tensor = input_tensor.value
    ttl_weight = weight.value
    ttl_output_tensor = ttl.tensor.rmsnorm(ttl_input_tensor, epsilon, ttl_weight)

    output_tensor = ttnn.Tensor(ttl_output_tensor)
    output_tensor = ttnn.reshape(output_tensor, original_shape)

    return output_tensor


def _torch_group_norm(input_tensor: ttnn.Tensor, *, num_groups, epsilon=1e-05, weight=None, bias=None, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    if weight is not None:
        weight = ttnn.from_device(weight)
        weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        weight = ttnn.to_torch(weight)
        if len(weight.shape) == 2:
            weight = weight[0]

    if bias is not None:
        bias = ttnn.from_device(bias)
        bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.to_torch(bias)
        if len(bias.shape) == 2:
            bias = bias[0]

    return torch.nn.functional.group_norm(input_tensor, num_groups, weight, bias, eps=epsilon)


@ttnn.decorate_operation(torch_function=_torch_group_norm)
def group_norm(
    input_tensor: ttnn.Tensor,
    *,
    num_groups: int,
    epsilon: float = 1e-12,
    weight: Optional[ttnn.Tensor] = None,
    bias: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    r"""
    group_norm(input_tensor: ttnn.Tensor) -> ttnn.Tensor

    Compute group_norm over :attr:`input_tensor`.

    """

    if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
        raise RuntimeError("group_norm expects input tensor to be on device!")

    output = _torch_group_norm(input_tensor, num_groups=num_groups, epsilon=epsilon, weight=weight, bias=bias)
    return ttnn.from_torch(output, dtype=input_tensor.dtype, layout=input_tensor.layout, device=input_tensor.device)


__all__ = [
    "layer_norm",
    "rms_norm",
    "group_norm",
]
