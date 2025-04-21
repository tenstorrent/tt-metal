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
Conv2dSliceConfig = ttnn._ttnn.operations.conv.Conv2dSliceConfig
Conv2dSliceHeight = ttnn._ttnn.operations.conv.Conv2dSliceConfig.SliceTypeEnum.SliceHeight
Conv2dSliceWidth = ttnn._ttnn.operations.conv.Conv2dSliceConfig.SliceTypeEnum.SliceWidth

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
    has_bias,
    groups,
    device,
    conv_config=None,
):
    """
    TTNN Conv2D applies preprocessing to the weights tensors before performing the convolution operation, to convert the weights into a format suitable for the operation.
    This can be applied just once to the weights and bias tensors, and the resulting tensors can be reused for multiple invocations of the same convolution operation.
    The exact format of the weights and bias tensors depends on the input tensor parameters and the sharding scheme.

    :param ttnn.Tensor weight_tensor: the weight tensor in PyTorch Conv2d format.
    :param ttnn.MemoryConfig input_memory_config: the memory configuration for the input tensor.
    :param ttnn.Tensor input_layout: the layout of the input tensor.
    :param ttnn.Tensor weights_format: the format of the weights tensor. Currently only supports OIHW. (out_channels, in_channels, kernel_height, kernel_width)
    :param int: in_channels:  number of input channels.
    :param int: out_channels:  number of output channels.
    :param int: batch_size:  batch size.
    :param int: input_height:  height of the input tensor.
    :param int: input_width:  width of the input tensor.
    :param tuple[int  , int] kernel_size: size of the convolving kernel.
    :param tuple[int, int] stride: stride of the cross-correlation.
    :param tuple[int, int] or tuple[int, int, int, int]) padding: zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
    :param tuple[int, int] dilation: spacing between kernel elements.
    :param bool has_bias:  whether the convolution has a bias term.
    :param int groups:  number of blocked connections from input channels to output channels.
    :param ttnn.Conv2dConfig, None conv_config: configuration for convolution. Default: None
    :param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None

    :return: The preprocessed weight tensor on device
    :rtype: [ttnn.Tensor]: The preprocessed bias tensor on device
    """
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
        has_bias=has_bias,
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
    """
    TTNN Conv2D applies preprocessing to the bias tensors before performing the convolution operation, to convert the bias into a format suitable for the operation.
    This can be applied just once to the weights and bias tensors, and the resulting tensors can be reused for multiple invocations of the same convolution operation.
    The exact format of the weights and bias tensors depends on the input tensor parameters and the sharding scheme.

    :param ttnn.Tensor bias: the bias tensor in PyTorch Conv2d format.
    :param ttnn.MemoryConfig input_memory_config: the memory configuration for the input tensor.
    :param ttnn.Tensor input_layout: the layout of the input tensor.
    :param ttnn.Tensor weights_format: the format of the weights tensor. Currently only supports OIHW. (out_channels, in_channels, kernel_height, kernel_width)
    :param int: in_channels:  number of input channels.
    :param int: out_channels:  number of output channels.
    :param int: batch_size:  batch size.
    :param int: input_height:  height of the input tensor.
    :param int: input_width:  width of the input tensor.
    :param tuple[int  , int] kernel_size: size of the convolving kernel.
    :param tuple[int, int] stride: stride of the cross-correlation.
    :param tuple[int, int] or tuple[int, int, int, int]) padding: zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
    :param tuple[int, int] dilation: spacing between kernel elements.
    :param ttnn.IDevice device:  the device to use.
    :param int groups:  number of blocked connections from input channels to output channels.
    :param ttnn.Conv2dConfig, None conv_config: configuration for convolution. Default: None
    :param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None

    :return: The preprocessed bias tensor on device
    :rtype: [ttnn.Tensor]: The preprocessed bias tensor on device

    """
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
    padding: Union[Tuple[int, int], Tuple[int, int, int, int]],
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    bias_tensor: ttnn.Tensor = None,
    conv_config: Conv2dConfig = None,  # config overrides by user
    compute_config=None,  # compute config overrides by user
    memory_config: ttnn.MemoryConfig = None,  # memory config overrides by user
    slice_config: Conv2dSliceConfig = None,  # slice config overrides by user
    return_output_dim=False,
    return_weights_and_bias=False,
) -> Tuple[ttnn.Tensor, Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]]:
    """
    Applies a 2D convolution over an input signal composed of several input planes.

    For more information, refer to `this tech report. <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/ttcnn.md>`_

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
    :param bool return_output_dim:  If true, the op also returns the height and width of the output tensor in [N, H, W, C] format,
    :param bool return_weights_and_bias:  If true, the op also returns the preprocessed weight and bias on device .

    :return: The output tensor, output height and width, and the preprocessed weights and bias.

    :rtype: [ttnn.Tensor]: The output tensor, when return_output_dim = False and return_weights_and_bias = False
    :rtype: [ttnn.Tensor, Tuple[int, int]]: The output tensor, and it's height and width, if return_output_dim = True
    :rtype: [ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and it's height and width, if return_weights_and_bias = True
    :rtype: [ttnn.Tensor, Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]]: The output tensor, and it's height and width, if return_output_dim = True and return_weights_and_bias = True

    """
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
        slice_config=slice_config,
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
