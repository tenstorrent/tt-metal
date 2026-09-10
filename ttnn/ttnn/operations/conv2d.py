# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from typing import Tuple, Union, Dict, Optional
import warnings
import math
import ttnn
from ttnn.operations.golden_common import golden_apply_fused_activations, golden_assemble_conditional_result

SlidingWindowParallelConfig = ttnn._ttnn.operations.sliding_window.ParallelConfig
Conv2dConfig = ttnn._ttnn.operations.conv.Conv2dConfig
PaddingMode = ttnn._ttnn.operations.conv.PaddingMode

# TODO: Remove Conv2dSliceConfig and update all relevant models & tests
Conv2dSliceConfig = ttnn._ttnn.operations.sliding_window.Op2DSliceConfig
Conv2dDRAMSliceHeight = Conv2dSliceConfig.SliceTypeEnum.DRAMSliceHeight
Conv2dDRAMSliceWidth = Conv2dSliceConfig.SliceTypeEnum.DRAMSliceWidth
Conv2dL1Full = Conv2dSliceConfig.SliceTypeEnum.L1Full
Conv2dL1FullSliceConfig = Conv2dSliceConfig(slice_type=Conv2dL1Full)

Op2DSliceConfig = ttnn._ttnn.operations.sliding_window.Op2DSliceConfig
Op2DDRAMSliceHeight = Op2DSliceConfig.SliceTypeEnum.DRAMSliceHeight
Op2DDRAMSliceWidth = Op2DSliceConfig.SliceTypeEnum.DRAMSliceWidth
Op2DL1Full = Op2DSliceConfig.SliceTypeEnum.L1Full
Op2DL1FullSliceConfig = Op2DSliceConfig(slice_type=Op2DL1Full)


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
    :param tuple[int, int] output_padding: additional size added to one side of each dimension in the output. Must match the value passed to conv_transpose2d so weight preparation and the op agree on the output size and slicing. Default: [0, 0].
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


def _normalize_pair(value):
    if hasattr(value, "__len__"):
        if len(value) != 2:
            raise ValueError("Expected a scalar or a sequence of 2 elements")
        return tuple(value)
    return (value, value)


def _normalize_conv_padding(padding):
    if not hasattr(padding, "__len__"):
        return (padding, padding, padding, padding)
    if len(padding) == 2:
        return (padding[0], padding[0], padding[1], padding[1])
    if len(padding) == 4:
        return tuple(padding)
    raise ValueError("Padding should be a scalar or a sequence of 2 or 4 elements")


def _reshape_conv_input_to_nchw(input_tensor, batch_size, input_height, input_width, in_channels):
    # 1, 1, NHW, C -> N, C, H, W
    return input_tensor.reshape(batch_size, input_height, input_width, -1)[..., :in_channels].permute(0, 3, 1, 2)


def _flatten_conv_output_to_ttnn_layout(output_tensor):
    batch_size, out_channels, output_height, output_width = output_tensor.shape
    # N, C, H, W -> 1, 1, NHW, C
    return output_tensor.permute(0, 2, 3, 1).reshape(1, 1, batch_size * output_height * output_width, out_channels)


def _flatten_conv_bias(bias_tensor):
    # Torch convolution functions expect a one-dimensional bias.
    return None if bias_tensor is None else bias_tensor.reshape(-1).float()


def _conv_activation(conv_config):
    return None if conv_config is None else conv_config.activation


def _processed_weights_and_bias_reference(weight_tensor, bias_tensor):
    weight_reference = weight_tensor.clone()
    ttnn.decorators.set_golden_comparison_config(weight_reference, method="skip", scope="all")
    if bias_tensor is None:
        bias_reference = None
    else:
        bias_reference = bias_tensor.clone()
        ttnn.decorators.set_golden_comparison_config(bias_reference, method="skip", scope="all")
    return weight_reference, bias_reference


def _assemble_conv_result(
    output_tensor,
    *,
    output_dim,
    weight_tensor,
    bias_tensor,
    return_output_dim,
    return_weights_and_bias,
):
    weights_and_bias = (
        _processed_weights_and_bias_reference(weight_tensor, bias_tensor) if return_weights_and_bias else None
    )
    return golden_assemble_conditional_result(
        output_tensor,
        (return_output_dim, output_dim),
        (return_weights_and_bias, weights_and_bias),
    )


def _golden_function_conv2d(
    input_tensor,
    weight_tensor,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_height: int,
    input_width: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    bias_tensor=None,
    conv_config: Conv2dConfig = None,
    return_output_dim=False,
    return_weights_and_bias=False,
    **_,
):
    import torch

    input_tensor = _reshape_conv_input_to_nchw(
        input_tensor,
        batch_size,
        input_height,
        input_width,
        in_channels,
    )

    pad_top, pad_bottom, pad_left, pad_right = _normalize_conv_padding(padding)

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
        bias=_flatten_conv_bias(bias_tensor),
        stride=stride,
        padding=(0, 0),
        dilation=dilation,
        groups=groups,
    )

    output_tensor = golden_apply_fused_activations(output_tensor, _conv_activation(conv_config))
    output_height, output_width = output_tensor.shape[-2:]
    output_tensor = _flatten_conv_output_to_ttnn_layout(output_tensor)
    return _assemble_conv_result(
        output_tensor,
        output_dim=(output_height, output_width),
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        return_output_dim=return_output_dim,
        return_weights_and_bias=return_weights_and_bias,
    )


ttnn.attach_golden_function(
    ttnn.conv2d,
    golden_function=_golden_function_conv2d,
)


def _golden_function_conv1d(
    input_tensor,
    weight_tensor,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_length: int,
    kernel_size: int,
    stride: int = 1,
    padding=0,
    dilation: int = 1,
    groups: int = 1,
    bias_tensor=None,
    conv_config: Conv2dConfig = None,
    return_output_dim=False,
    return_weights_and_bias=False,
    **_,
):
    import torch

    input_tensor = input_tensor.reshape(batch_size, input_length, -1)[..., :in_channels].permute(0, 2, 1)
    if weight_tensor.ndim == 4:
        if weight_tensor.shape[-2] != 1:
            raise ValueError("Conv1d 4D weights must have a singleton kernel-height dimension")
        torch_weight = weight_tensor.squeeze(-2)
    else:
        torch_weight = weight_tensor

    if hasattr(padding, "__len__"):
        if len(padding) != 2:
            raise ValueError("Conv1d padding should be a scalar or a sequence of 2 elements")
        pad_left, pad_right = padding
    else:
        pad_left = pad_right = padding
    padded_input = torch.nn.functional.pad(input_tensor.float(), (pad_left, pad_right))
    output_tensor = torch.nn.functional.conv1d(
        padded_input,
        torch_weight.float(),
        bias=_flatten_conv_bias(bias_tensor),
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=groups,
    )
    output_tensor = golden_apply_fused_activations(output_tensor, _conv_activation(conv_config))
    output_length = output_tensor.shape[-1]
    output_tensor = output_tensor.permute(0, 2, 1).reshape(1, 1, batch_size * output_length, out_channels)
    return _assemble_conv_result(
        output_tensor,
        output_dim=output_length,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        return_output_dim=return_output_dim,
        return_weights_and_bias=return_weights_and_bias,
    )


ttnn.attach_golden_function(
    ttnn.conv1d,
    golden_function=_golden_function_conv1d,
)


def _crop_transposed_convolution_output(output_tensor, padding):
    pad_top, pad_bottom, pad_left, pad_right = padding
    height_end = output_tensor.shape[-2] - pad_bottom if pad_bottom else None
    width_end = output_tensor.shape[-1] - pad_right if pad_right else None
    return output_tensor[..., pad_top:height_end, pad_left:width_end]


def _golden_function_conv_transpose2d(
    input_tensor,
    weight_tensor,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_height: int,
    input_width: int,
    kernel_size,
    stride=(1, 1),
    padding=(0, 0),
    output_padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
    bias_tensor=None,
    conv_config: Conv2dConfig = None,
    mirror_kernel=True,
    return_output_dim=False,
    return_weights_and_bias=False,
    **_,
):
    import torch

    input_tensor = _reshape_conv_input_to_nchw(
        input_tensor,
        batch_size,
        input_height,
        input_width,
        in_channels,
    )
    torch_weight = weight_tensor if mirror_kernel else torch.flip(weight_tensor, dims=(-2, -1))
    output_tensor = torch.nn.functional.conv_transpose2d(
        input_tensor.float(),
        torch_weight.float(),
        bias=_flatten_conv_bias(bias_tensor),
        stride=_normalize_pair(stride),
        padding=0,
        output_padding=_normalize_pair(output_padding),
        groups=groups,
        dilation=_normalize_pair(dilation),
    )
    output_tensor = _crop_transposed_convolution_output(output_tensor, _normalize_conv_padding(padding))
    output_tensor = golden_apply_fused_activations(output_tensor, _conv_activation(conv_config))
    output_height, output_width = output_tensor.shape[-2:]
    output_tensor = _flatten_conv_output_to_ttnn_layout(output_tensor)
    return _assemble_conv_result(
        output_tensor,
        output_dim=(output_height, output_width),
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        return_output_dim=return_output_dim,
        return_weights_and_bias=return_weights_and_bias,
    )


ttnn.attach_golden_function(
    ttnn.conv_transpose2d,
    golden_function=_golden_function_conv_transpose2d,
)

__all__ = []
