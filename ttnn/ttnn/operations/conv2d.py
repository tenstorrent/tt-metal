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


Conv2dConfig = ttnn._ttnn.operations.conv2d.Conv2dConfig

get_conv_padded_input_shape_and_mem_config = ttnn._ttnn.operations.conv2d.get_conv_padded_input_shape_and_mem_config
OptimizedConvParallelizationConfig = ttnn._ttnn.operations.conv2d.OptimizedConvParallelizationConfig
OptimizedConvBlockConfig = ttnn._ttnn.operations.conv2d.OptimizedConvBlockConfig


def get_conv_output_dim(input, window, stride=1, pad=0, dilation=1):
    """
    Returns the output dimension of a convolution operation.
    """
    return (input + (2 * pad) - dilation * (window - 1) - 1) // stride + 1


def prepare_conv_weights_for_ttnn(
    *,
    weight_tensor,
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
    return ttnn._ttnn.operations.conv2d.prepare_conv_weights_for_ttnn(
        weight_tensor=weight_tensor,
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


def prepare_conv_bias_for_ttnn(
    *,
    bias_tensor,
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
    return ttnn._ttnn.operations.conv2d.prepare_conv_bias_for_ttnn(
        bias_tensor=bias_tensor,
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
    return ttnn._ttnn.operations.conv2d.convert_conv_weight_tensor_to_tiled_layout(
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
    return ttnn._ttnn.operations.conv2d.convert_conv_weight_tensor_to_special_padding_tiled_layout(
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
    return ttnn._ttnn.operations.conv2d.convert_conv_weight_tensor_to_grouped_layout(
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
    memory_config: ttnn.MemoryConfig = None,  # memory config overrides by user
    conv_op_cache={},  # basic conv object caching in python needed for intermediate refactoring. Not needed after full op refactoring in C++.
    debug=False,  # ignored
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
    return ttnn._ttnn.operations.conv2d.conv2d_host_weights(
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
        memory_config=memory_config,
    )


@ttnn.register_python_operation(name="ttnn.conv2d_device_weights")
def conv2d_device_weights(
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
    memory_config: ttnn.MemoryConfig = None,  # memory config overrides by user
    conv_op_cache={},  # basic conv object caching in python needed for intermediate refactoring. Not needed after full op refactoring in C++.
    debug=False,  # ignored
) -> ttnn.Tensor:
    return ttnn._ttnn.operations.conv2d.conv2d(
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
        memory_config=memory_config,
    )


__all__ = []
