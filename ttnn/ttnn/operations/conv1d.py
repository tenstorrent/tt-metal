# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict, Optional
import torch
import warnings
import math
import ttnn

Conv1dConfig = ttnn._ttnn.operations.conv.Conv2dConfig


@ttnn.register_python_operation(name="ttnn.conv1d")
def Conv1d(
    *,
    input_tensor: ttnn.Tensor,  # may or may not be sharded
    weight_tensor: ttnn.Tensor,
    device: ttnn.Device,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias_tensor: ttnn.Tensor = None,
    conv_config: Conv1dConfig = None,  # config overrides by user
    compute_config: ttnn.DeviceComputeKernelConfig = None,
    return_output_dim=False,
    return_weights_and_bias=False,
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
    """
    Applies a 1D convolution over an input signal composed of several input planes. Implemented as a 2D Convolution of input height 1 and input width as input_length.

        :param ttnn.Tensor input_tensor:  The input tensor. This must be in the format [N, H, W, C]. It can be on host or device.
        :param ttnn.Tensor weight_tensor: The weight tensor. The weights can be passed in the same format as PyTorch, [out_channels, in_channels, kernel_height, kernel_width]. The op w
        :param ttnn.Tensor, None bias_tensor:   Optional bias tensor. Default: None
        :param ttnn.IDevice device:  The device to use.
        :param int: in_channels:  Number of input channels.
        :param int: out_channels:  Number of output channels.
        :param int: batch_size:  Batch size.
        :param int: input_height:  Height of the input tensor.
        :param int: input_width:  Width of the input tensor.
        :param tuple[int  , int] kernel_size: Size of the convolving kernel.
        :param tuple[int, int] stride: Stride of the cross-correlation.
        :param tuple[int, int] or tuple[int, int, int, int]) padding: Zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
        :param tuple[int, int] dilation: Spacing between kernel elements.
        :param int groups:  Number of blocked connections from input channels to output channels.
        :param ttnn.Conv2dConfig, None conv_config: Configuration for convolution. Default: None
        :param ttnn.DeviceComputeKernelConfig, None compute_config: Configuration for compute kernel. Default: None
        :param ttnn.MemoryConfig, None memory_config: Output Tensor's Memory Configuration. Default: None
        :param bool return_output_dim:  If true, the op also returns the height & width of the output tensor in [N, H, W, C] format,
        :param bool return_weights_and_bias:  If true, the op also returns the preprocessed weight and bias on device .

        :return: The output tensor, output height & width, and the preprocessed weights & bias.

        :rtype: [ttnn.Tensor]: The output tensor, when return_output_dim = False and return_weights_and_bias = False
        :rtype: [ttnn.Tensor, Tuple[int, int]]: The output tensor, and it's height & width, if return_output_dim = True
        :rtype: [ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and it's height & width, if return_weights_and_bias = True
        :rtype: [ttnn.Tensor, Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and it's height & width, if return_output_dim = True and return_weights_and_bias = True

    """
    # Reshape the input and weight tensors to 4D for conv2d operation
    # Should be no-op as input_tensor is in RM layout
    if len(input_tensor.shape) != 4:
        input_tensor = ttnn.reshape(input_tensor, [batch_size, input_length, 1, in_channels])
    if len(weight_tensor.shape) != 4:
        weight_tensor = ttnn.reshape(weight_tensor, [out_channels, in_channels // groups, kernel_size, 1])

    (
        output_tensor_new,
        output_length_new,
        _,
        weight_tensor_on_dev_new,
        bias_tensor_on_dev_new,
    ) = ttnn._ttnn.operations.conv.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_length,
        input_width=1,
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
        padding=(padding, 0),
        dilation=(dilation, 1),
        groups=groups,
        bias_tensor=bias_tensor,
        conv_config=conv_config,
        compute_config=compute_config,
    )

    if return_output_dim and return_weights_and_bias:
        return output_tensor_new, output_length_new, [weight_tensor_on_dev_new, bias_tensor_on_dev_new]
    elif return_weights_and_bias:
        return output_tensor_new, [weight_tensor_on_dev_new, bias_tensor_on_dev_new]
    elif return_output_dim:
        return output_tensor_new, output_length_new
    else:
        return output_tensor_new


__all__ = []
