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
    return_output_dim=False,
    return_weights_and_bias=False,
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

    if return_output_dim or return_weights_and_bias:
        return [output_tensor]

    return output_tensor


ttnn.attach_golden_function(
    ttnn.conv2d,
    golden_function=_golden_function,
)

__all__ = []
