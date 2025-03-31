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

Conv2dConfig = ttnn._ttnn.operations.conv.Conv2dConfig


@ttnn.register_python_operation(name="ttnn.conv_transpose2d")
def conv_transpose2d(
    *,
    input_tensor: ttnn.Tensor,  # may or may not be sharded
    weight_tensor: ttnn.Tensor,
    in_channels: int,
    out_channels: int,
    bias_tensor: ttnn.Tensor,
    device: ttnn.Device,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    output_padding: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
    batch_size: int,
    input_height: int,
    input_width: int,
    conv_config: Conv2dConfig = None,  # config overrides by user
    compute_config=None,  # compute config overrides by user
    groups: int = 1,
    mirror_kernel=True,
    return_output_dim=False,
    return_weights_and_bias=False,
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
    (
        conv_output,
        output_height,
        output_width,
        prepared_device_weight,
        prepared_device_bias,
    ) = ttnn._ttnn.operations.conv.conv_transpose2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        in_channels=in_channels,
        out_channels=out_channels,
        bias_tensor=bias_tensor,
        device=device,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        mirror_kernel=mirror_kernel,
    )

    if return_output_dim and return_weights_and_bias:
        return conv_output, [output_height, output_width], [prepared_device_weight, prepared_device_bias]
    elif return_weights_and_bias:
        return conv_output, [prepared_device_weight, prepared_device_bias]
    elif return_output_dim:
        return conv_output, [output_height, output_width]
    else:
        return conv_output


__all__ = []
