# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from typing import Tuple, Union, Dict, Optional
import torch
import warnings
import math
import ttnn
from ttnn.device import (
    is_grayskull,
    is_wormhole_b0,
)


def _nearest_32(x):
    return math.ceil(x / 32) * 32


Conv2dConfig = ttnn._ttnn.operations.conv.Conv2dConfig

OptimizedConvParallelizationConfig = ttnn._ttnn.operations.conv.OptimizedConvParallelizationConfig
OptimizedConvBlockConfig = ttnn._ttnn.operations.conv.OptimizedConvBlockConfig


def get_conv_output_dim(input, window, stride=1, pad=0, dilation=1):
    """
    Returns the output dimension of a convolution operation.
    """
    return (input + (2 * pad) - dilation * (window - 1) - 1) // stride + 1


def prepare_conv_weights(
    *,
    weight_tensor,
    input_memory_config,
    input_layout,
    weights_format,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    device,
    conv_config=None,
):
    return ttnn._ttnn.operations.conv.prepare_conv_weights(
        weight_tensor=weight_tensor,
        input_memory_config=input_memory_config,
        input_tensor_layout=input_layout,
        weights_format=weights_format,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=list(kernel_size),
        stride=list(stride),
        padding=list(padding),
        dilation=list(dilation),
        groups=groups,
        device=device,
        conv_config=conv_config,
    )


def prepare_conv_bias(
    *,
    bias_tensor,
    input_memory_config,
    input_layout,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    device,
    conv_config=None,
):
    return ttnn._ttnn.operations.conv.prepare_conv_bias(
        bias_tensor=bias_tensor,
        input_memory_config=input_memory_config,
        input_tensor_layout=input_layout,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=list(kernel_size),
        stride=list(stride),
        padding=list(padding),
        dilation=list(dilation),
        groups=groups,
        device=device,
        conv_config=conv_config,
    )


def convert_conv_weight_tensor_to_tiled_layout(conv_weight_tensor, in1_block_h, in1_block_w, output_dtype=None):
    """
    Converts convolution weights to 2d matrix tiled layout on host
    Returns a new tensor with the converted layout.

    +----------+----------------------+-----------+-------------+----------+
    | Argument | Description          | Data type | Valid range | Required |
    +==========+======================+===========+=============+==========+
    | a        | Input tensor         | Tensor    |             | Yes      |
    +----------+----------------------+-----------+-------------+----------+
    """
    return ttnn._ttnn.operations.conv.convert_conv_weight_tensor_to_tiled_layout(
        conv_weight_tensor, in1_block_h, in1_block_w, output_dtype
    )


def convert_conv_weight_tensor_to_special_padding_tiled_layout(
    conv_weight_tensor, in1_block_h, in1_block_w, output_dtype=None
):
    """
    Converts convolution weights to 2d matrix tiled layout on host with special block height padding
    Returns a new tensor with the converted layout.

    +----------+----------------------+-----------+-------------+----------+
    | Argument | Description          | Data type | Valid range | Required |
    +==========+======================+===========+=============+==========+
    | a        | Input tensor         | Tensor    |             | Yes      |
    +----------+----------------------+-----------+-------------+----------+
    """
    return ttnn._ttnn.operations.conv.convert_conv_weight_tensor_to_special_padding_tiled_layout(
        conv_weight_tensor, in1_block_h, in1_block_w, output_dtype
    )


def convert_conv_weight_tensor_to_grouped_layout(conv_weight_tensor, num_groups, output_dtype):
    """
    Converts convolution weights to grouped layout with padded zeros
    Returns a new tensor with the converted layout.

    +----------+----------------------+-----------+-------------+----------+
    | Argument | Description          | Data type | Valid range | Required |
    +==========+======================+===========+=============+==========+
    | a        | Input tensor         | Tensor    |             | Yes      |
    +----------+----------------------+-----------+-------------+----------+
    """
    return ttnn._ttnn.operations.conv.convert_conv_weight_tensor_to_grouped_layout(
        conv_weight_tensor, num_groups, output_dtype
    )


@ttnn.register_python_operation(name="ttnn.conv2d")
def conv2d(
    *,
    input_tensor: ttnn.Tensor,  # may or may not be sharded
    weight_tensor: ttnn.Tensor,
    device: ttnn.Device,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_height: int,
    input_width: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    bias_tensor: ttnn.Tensor = None,
    conv_config: Conv2dConfig = None,  # config overrides by user
    compute_config=None,  # compute config overrides by user
    memory_config: ttnn.MemoryConfig = None,  # memory config overrides by user
    conv_op_cache={},  # basic conv object caching in python needed for intermediate refactoring. Not needed after full op refactoring in C++.
    debug=False,  # ignored
    return_output_dim=False,
    return_weights_and_bias=False,
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
    (
        conv_output,
        output_height,
        output_width,
        prepared_device_weight,
        prepared_device_bias,
    ) = ttnn._ttnn.operations.conv.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_tensor=bias_tensor,
        conv_config=conv_config,
        compute_config=compute_config,
        memory_config=memory_config,
    )

    if return_output_dim and return_weights_and_bias:
        return conv_output, [output_height, output_width], [prepared_device_weight, prepared_device_bias]
    elif return_weights_and_bias:
        return conv_output, [prepared_device_weight, prepared_device_bias]
    elif return_output_dim:
        return conv_output, [output_height, output_width]
    else:
        return conv_output


def get_activation_function(name: str):
    if name == "relu":
        return torch.nn.functional.relu
    elif name == "":
        return lambda x: x
    else:
        raise RuntimeError(f"Unexpected activation function: '{name}'")


def _golden_function(
    input_tensor,
    weight_tensor,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_height: int,
    input_width: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    bias_tensor=None,
    conv_config: Conv2dConfig = None,
    **_,
):
    import torch

    input_tensor = input_tensor.reshape(batch_size, input_height, input_width, -1)[:, :, :, :in_channels].permute(
        0, 3, 1, 2
    )  # 1, 1, NHW, C -> N, C, H, W

    bias_tensor = bias_tensor.reshape(-1)  # torch expected 1D bias

    output_tensor = torch.nn.functional.conv2d(
        input_tensor.float(),
        weight_tensor.float(),
        bias=bias_tensor.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    output_tensor = (
        get_activation_function(conv_config.activation)(output_tensor) if conv_config is not None else output_tensor
    )

    N, C, H, W = output_tensor.shape
    output_tensor = output_tensor.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)  # N, C, H, W -> 1, 1, NHW, C

    return [output_tensor]


ttnn.attach_golden_function(
    ttnn.conv2d,
    golden_function=_golden_function,
)

__all__ = []
