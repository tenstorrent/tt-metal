# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from typing import Tuple, Union, Dict, Optional
import warnings
import math
import ttnn
from ttnn.operations.activations import get_golden_function_for_activation

SlidingWindowParallelConfig = ttnn._ttnn.operations.sliding_window.ParallelConfig
Conv2dConfig = ttnn._ttnn.operations.conv.Conv2dConfig
Conv2dSliceConfig = ttnn._ttnn.operations.conv.Conv2dSliceConfig
Conv2dDRAMSliceHeight = ttnn._ttnn.operations.conv.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceHeight
Conv2dDRAMSliceWidth = ttnn._ttnn.operations.conv.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceWidth
Conv2dL1Full = ttnn._ttnn.operations.conv.Conv2dSliceConfig.SliceTypeEnum.L1Full
Conv2dL1FullSliceConfig = Conv2dSliceConfig(slice_type=Conv2dL1Full)


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

    # Get activation from conv_config
    activation = None
    if conv_config is not None:
        activation = conv_config.activation

    act_func = get_golden_function_for_activation(activation)
    output_tensor = act_func(output_tensor) if act_func is not None else output_tensor

    N, C, H, W = output_tensor.shape
    output_tensor = output_tensor.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)  # N, C, H, W -> 1, 1, NHW, C

    if return_output_dim or return_weights_and_bias:
        return [output_tensor]

    return output_tensor


def conv2d_unfold_matmul(input, weight, bias=None, stride=1, padding=0, dilation=1, matmul_precision="highest"):
    """
    Recreate torch.conv2d using unfold and matmul operations.

    This function provides an alternative implementation of 2D convolution that decomposes
    the operation into im2col (unfold) followed by matrix multiplication. This is useful
    for understanding the relationship between convolution and matmul, and for comparing
    numerical precision across different implementations.

    Note: This implementation assumes groups=1 (standard convolution).

    Args:
        input (torch.Tensor): Input tensor of shape (N, C_in, H_in, W_in)
        weight (torch.Tensor): Weight tensor of shape (C_out, C_in, kH, kW)
        bias (torch.Tensor, optional): Bias tensor of shape (C_out,). Default: None
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int or tuple): Zero-padding added to both sides of input. Default: 0
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        matmul_precision (str): Torch matmul precision, either "highest", "high", or "medium". Default: "highest"
            - "highest": Most accurate, uses TF32 on Ampere+ GPUs (mantissa: ~19 bits)
            - "medium": Faster but less accurate, uses TF32 (mantissa: 10 bits)
            - "high": Balance between speed and accuracy (mantissa: ~16 bits)

    Returns:
        torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out)

    Example:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> weight = torch.randn(64, 3, 3, 3)
        >>> # Standard conv2d
        >>> output1 = torch.nn.functional.conv2d(input, weight, stride=1, padding=1)
        >>> # Unfold + matmul version with highest precision
        >>> output2 = conv2d_unfold_matmul(input, weight, stride=1, padding=1, matmul_precision="highest")
        >>> # outputs should be very close (within numerical precision)

    Note:
        This implementation is primarily for testing and analysis. For production use,
        torch.nn.functional.conv2d is more optimized.
    """
    import torch

    # Ensure stride, padding, and dilation are tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Set torch matmul precision
    original_precision = torch.get_float32_matmul_precision()
    if matmul_precision in ["highest", "high", "medium"]:
        torch.set_float32_matmul_precision(matmul_precision)
    else:
        raise ValueError(f"matmul_precision must be 'highest', 'high', or 'medium', got {matmul_precision}")

    try:
        # Extract dimensions
        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape

        # Calculate output dimensions
        out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
        out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1

        # Step 1: Unfold - Extract sliding local blocks (im2col operation)
        # Output shape: (N, C_in * kH * kW, L) where L = out_height * out_width
        unfolded = torch.nn.functional.unfold(
            input, kernel_size=(kernel_height, kernel_width), dilation=dilation, padding=padding, stride=stride
        )
        # unfolded shape: (batch_size, in_channels * kernel_height * kernel_width, out_height * out_width)

        # Step 2: Reshape weight for matmul
        # Weight shape: (out_channels, in_channels, kernel_height, kernel_width)
        # Reshape to: (out_channels, in_channels * kernel_height * kernel_width)
        weight_reshaped = weight.view(out_channels, -1)
        # weight_reshaped shape: (out_channels, in_channels * kernel_height * kernel_width)

        # Step 3: Perform matrix multiplication
        # (out_channels, K) @ (K, L) = (out_channels, L)
        # where K = in_channels * kernel_height * kernel_width, L = out_height * out_width
        output = torch.matmul(weight_reshaped, unfolded)
        # output shape: (batch_size, out_channels, out_height * out_width)

        # Step 4: Reshape output to spatial dimensions
        output = output.view(batch_size, out_channels, out_height, out_width)

        # Step 5: Add bias if provided
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)

        return output

    finally:
        # Restore original precision
        torch.set_float32_matmul_precision(original_precision)


ttnn.attach_golden_function(
    ttnn.conv2d,
    golden_function=_golden_function,
)

__all__ = []
