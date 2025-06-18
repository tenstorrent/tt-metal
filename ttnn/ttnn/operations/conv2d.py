# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from typing import Tuple, Union, Dict, Optional
import warnings
import math
import ttnn

SlidingWindowParallelConfig = ttnn._ttnn.operations.sliding_window.ParallelConfig
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


# TODO: remove this function after #21040 is fixed
def prepare_conv_transpose2d_weights(*args, **kwargs):
    """
    TTNN ConvTranspose2D applies preprocessing to the weights tensors before performing the conv_tranpose2D operation, to convert the weights into a format suitable for the operation.
    This can be applied just once to the weights and bias tensors, and the resulting tensors can be reused for multiple invocations of the same operation.
    The exact format of the weights and bias tensors depends on the input tensor parameters and the sharding scheme.

    :param ttnn.Tensor weight_tensor: the weight tensor in PyTorch Conv2d format.
    :param ttnn.MemoryConfig input_memory_config: the memory configuration for the input tensor.
    :param ttnn.Tensor input_layout: the layout of the input tensor.
    :param ttnn.Tensor weights_format: the format of the weights tensor. Currently only supports IOHW. (in_channels, out_channels, kernel_height, kernel_width)
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
    :param ttnn.DataType input_dtype: the data type of the input tensor.
    :param ttnn.DataType, None output_dtype: the data type of the output tensor. Default None (uses input_dtype)
    :param ttnn.Conv2dConfig, None conv_config: configuration for convolution. Default: None
    :param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None

    :return: The preprocessed weight tensor on device
    :rtype: [ttnn.Tensor]: The preprocessed bias tensor on device
    """
    return ttnn._ttnn.operations.conv.prepare_conv_transpose2d_weights(*args, **kwargs)


# TODO: remove this function after #21040 is fixed
def prepare_conv_transpose2d_bias(*args, **kwargs):
    """
    TTNN ConvTranspose2D applies preprocessing to the bias tensors before performing the convolution operation, to convert the bias into a format suitable for the operation.
    This can be applied just once to the weights and bias tensors, and the resulting tensors can be reused for multiple invocations of the same convolution operation.
    The exact format of the weights and bias tensors depends on the input tensor parameters and the sharding scheme.

    :param ttnn.Tensor bias: the bias tensor in PyTorch Conv2d format.
    :param ttnn.MemoryConfig input_memory_config: the memory configuration for the input tensor.
    :param ttnn.Tensor input_layout: the layout of the input tensor.
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
    :param ttnn.DataType input_dtype: the data type of the input tensor.
    :param ttnn.DataType, None output_dtype: the data type of the output tensor. Default None (uses input_dtype)
    :param int groups:  number of blocked connections from input channels to output channels.
    :param ttnn.Conv2dConfig, None conv_config: configuration for convolution. This config must have weights_dtype set to the same dtype as the processed weights tensor. Default: None
    :param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None

    :return: The preprocessed bias tensor on device
    :rtype: [ttnn.Tensor]: The preprocessed bias tensor on device

    """
    return ttnn._ttnn.operations.conv.prepare_conv_transpose2d_bias(*args, **kwargs)


# TODO: remove this function after #21040 is fixed
def prepare_conv_weights(*args, **kwargs):
    """
    TTNN Conv2D applies preprocessing to the weights tensors before performing the convolution operation, to convert the weights into a format suitable for the operation.
    This can be applied just once to the weights and bias tensors, and the resulting tensors can be reused for multiple invocations of the same convolution operation.
    The exact format of the weights and bias tensors depends on the input tensor parameters and the sharding scheme.

    :param ttnn.Tensor weight_tensor: the weight tensor in PyTorch Conv2d format.
    :param ttnn.MemoryConfig input_memory_config: the memory configuration for the input tensor.
    :param ttnn.Tensor input_layout: the layout of the input tensor.
    :param ttnn.Tensor weights_format: the format of the weights tensor. Currently only supports OIHW. (out_channels, in_channels, kernel_height, kernel_width)
    :param int in_channels:  number of input channels.
    :param int out_channels:  number of output channels.
    :param int batch_size:  batch size.
    :param int input_height:  height of the input tensor.
    :param int input_width:  width of the input tensor.
    :param tuple[int, int] kernel_size: size of the convolving kernel.
    :param tuple[int, int] stride: stride of the cross-correlation.
    :param tuple[int, int] or tuple[int, int, int, int]) padding: zero-padding added to both sides of the input. [pad_height, pad_width] or [pad_top, pad_bottom, pad_left, pad_right].
    :param tuple[int, int] dilation: spacing between kernel elements.
    :param bool has_bias:  whether the convolution has a bias term.
    :param int groups:  number of blocked connections from input channels to output channels.
    :param ttnn.DataType input_dtype: the data type of the input tensor.
    :param ttnn.DataType, None output_dtype: the data type of the output tensor. Default None (uses input_dtype)
    :param ttnn.Conv2dConfig, None conv_config: configuration for convolution. Default: None
    :param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None

    :return: The preprocessed weight tensor on device
    :rtype: [ttnn.Tensor]: The preprocessed bias tensor on device
    """
    return ttnn._ttnn.operations.conv.prepare_conv_weights(*args, **kwargs)


# TODO: remove this function after #21040 is fixed
def prepare_conv_bias(*args, **kwargs):
    """
    TTNN Conv2D applies preprocessing to the bias tensors before performing the convolution operation, to convert the bias into a format suitable for the operation.
    This can be applied just once to the weights and bias tensors, and the resulting tensors can be reused for multiple invocations of the same convolution operation.
    The exact format of the weights and bias tensors depends on the input tensor parameters and the sharding scheme.

    :param ttnn.Tensor bias: the bias tensor in PyTorch Conv2d format.
    :param ttnn.MemoryConfig input_memory_config: the memory configuration for the input tensor.
    :param ttnn.Tensor input_layout: the layout of the input tensor.
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
    :param ttnn.DataType input_dtype: the data type of the input tensor.
    :param ttnn.DataType, None output_dtype: the data type of the output tensor. Default None (uses input_dtype)
    :param int groups:  number of blocked connections from input channels to output channels.
    :param ttnn.Conv2dConfig, None conv_config: configuration for convolution. This config must have weights_dtype set to the same dtype as the processed weights tensor. Default: None
    :param ttnn.DeviceComputeKernelConfig, None compute_config: configuration for compute kernel. Default: None

    :return: The preprocessed bias tensor on device
    :rtype: [ttnn.Tensor]: The preprocessed bias tensor on device

    """
    return ttnn._ttnn.operations.conv.prepare_conv_bias(*args, **kwargs)


def get_torch_act_func_from_string(act_string):
    import torch

    act_func_map = {
        "relu": torch.nn.functional.relu,
        "silu": torch.nn.functional.silu,
        "mish": torch.nn.functional.mish,
        "sigmoid": torch.nn.functional.sigmoid,
        "sigmoid_approx": torch.nn.functional.sigmoid,
        "tanh": torch.nn.functional.tanh,
        "log": torch.log,
        "softplus": torch.nn.functional.softplus,
        "gelu": torch.nn.functional.gelu,
        "sqrt": torch.sqrt,
    }
    if act_string == "":
        return None
    if act_string in act_func_map:
        return act_func_map[act_string]
    raise RuntimeError(f"Activation function {act_string} not supported")


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
    padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
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

    if hasattr(padding, "__len__"):
        if len(padding) == 2:
            pad_top = padding[0]
            pad_bottom = padding[0]
            pad_left = padding[1]
            pad_right = padding[1]
        elif len(padding) == 4:
            pad_top = padding[0]
            pad_bottom = padding[1]
            pad_left = padding[2]
            pad_right = padding[3]
        else:
            raise ValueError("Padding should be a scalar or a list of 2 or 4 elements")
    else:
        pad_top = padding
        pad_bottom = padding
        pad_left = padding
        pad_right = padding

    # this is done because torch doesn't support different padding for height and width (e.g. padding = (1, 2, 3, 4))
    torch_padded_input = torch.nn.functional.pad(
        input_tensor.float(),
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )

    # padding is (0, 0) because the padding is already applied to the input tensor above
    output_tensor = torch.nn.functional.conv2d(
        torch_padded_input,
        weight_tensor.float(),
        bias=bias_tensor.float(),
        stride=stride,
        padding=(0, 0),
        dilation=dilation,
        groups=groups,
    )

    act_func = get_torch_act_func_from_string(conv_config.activation) if conv_config is not None else None
    output_tensor = act_func(output_tensor) if act_func is not None else output_tensor

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
