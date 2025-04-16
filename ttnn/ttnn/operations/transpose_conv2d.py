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
    """
    Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.

        :param ttnn.Tensor input_tensor:  the input tensor.
        :param ttnn.Tensor weight_tensor: the weight tensor.
        :param ttnn.Tensor, None bias_tensor:   optional bias tensor. Default: None
        :param ttnn.IDevice device:  the device to use.
        :param int: in_channels:  number of input channels.
        :param int: out_channels:  number of output channels.
        :param int: batch_size:  batch size.
        :param int: input_height:  height of the input tensor.
        :param int: input_width:  width of the input tensor.
        :param tuple[int  , int] kernel_size: size of the convolving kernel.
        :param tuple[int, int] stride: stride of the cross-correlation.
        :param tuple[int, int] or tuple[int, int, int, int]) padding: zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
        :param tuple[int, int] dilation: spacing between kernel elements.
        :param int groups:  number of blocked connections from input channels to output channels.
        :param ttnn.Conv2dConfig, None conv_config: configuration for convolution. Default: None
        :param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None
        :param bool mirror_kernel: Determines if the op should mirror the kernel internally. Should be set to True if the kernel has already been mirrored.
        :param bool: return_output_dim:  If true, the op also returns the height and width of the output tensor in [N, H, W, C] format,
        :param bool: return_weights_and_bias:  If true, the op also returns the preprocessed weight and bias on device .

        :return: The output tensor, output height and width, and the preprocessed weights and bias.

        :rtype: [ttnn.Tensor]: the output tensor, when return_output_dim = False and return_weights_and_bias = False
        :rtype: [ttnn.Tensor, Tuple[int, int]]: the output tensor, and it's height and width, if return_output_dim = True
        :rtype: [ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and it's height and width, if return_weights_and_bias = True
        :rtype: [ttnn.Tensor, Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]]: the output tensor, and it's height and width, if return_output_dim = True and return_weights_and_bias = True
    """
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
