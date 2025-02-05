# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from typing import Tuple, Union, Dict, Optional, List
import torch
import warnings
import math
import ttnn
from ttnn.device import (
    is_grayskull,
    is_wormhole_b0,
)
import copy


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


def get_conv_input_dim(output, window, stride=1, pad=0, dilation=1):
    """
    Returns the input dimension required to achieve a given output dimension in a convolution operation.
    """
    return (output - 1) * stride + dilation * (window - 1) + 1 - 2 * pad


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


def calculate_conv_height_split_params(
    input_height: int,
    num_input_slices: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> List[Tuple[int, int, int, int]]:
    # if input is sliced into num_input_slices, so is output
    # we use output slices to figure out where are they coming from input image
    output_height = get_conv_output_dim(input_height, kernel_size, stride, padding, dilation)
    kernel_size_w_dilation = kernel_size + (kernel_size - 1) * (dilation - 1)
    output_values = []
    output_slice_height = output_height // num_input_slices
    for output_slice_height_start in range(0, output_height, output_slice_height):
        output_slice_height_end = output_slice_height_start + output_slice_height
        output_slice_height_end = min(output_slice_height_end, output_height)  # last slice may be smaller

        input_slice_height_start = output_slice_height_start * stride - padding
        input_slice_height_end = (output_slice_height_end - 1) * stride + kernel_size_w_dilation - padding
        pad_top = max(0, -input_slice_height_start)
        pad_bottom = max(0, input_slice_height_end - input_height)
        input_slice_height_start = max(0, input_slice_height_start)
        input_slice_height_end = min(input_height, input_slice_height_end)
        output_values.append((input_slice_height_start, input_slice_height_end, pad_top, pad_bottom))

    # TBD if we want this: merge last slice with previous it's less than H=32
    # if output_values[-1][1] - output_values[-1][0] < 32:
    #     output_values[-2] = (output_values[-2][0], output_values[-1][1], output_values[-2][2], output_values[-1][3])
    #     output_values.pop()
    return output_values, output_height


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


@ttnn.register_python_operation(name="ttnn.conv2d_sliced")
def conv2d_sliced(
    *,
    input_tensor: ttnn.Tensor,  # may or may not be sharded
    weight_tensor: ttnn.Tensor,
    device: ttnn.Device,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_height: int,
    input_width: int,
    num_input_slices: int,
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
    if num_input_slices <= 1:
        return conv2d(
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
            conv_op_cache=conv_op_cache,
            debug=debug,
            return_output_dim=return_output_dim,
            return_weights_and_bias=return_weights_and_bias,
        )

    assert groups == 1, "Groups != 1 is not supported for sliced conv2d t the moment"
    assert batch_size == 1, "Batch size != 1 is not supported for sliced conv2d at the moment"
    # What to do with output mem config?

    # Expect input tensor to be in [1, 1, N * H * W, C] format
    # At the moment, restore it back to [N, H, W, C] format
    input_tensor = input_tensor.reshape(batch_size, input_height, input_width, -1)

    # Need this for slice op
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        input_tensor = ttnn.to_device(input_tensor, device)

    conv_height_split_params, conv_output_height = calculate_conv_height_split_params(
        input_height, num_input_slices, kernel_size[0], stride[0], padding[0], dilation[0]
    )

    # Prepare conv config for slices
    # conv2d_config_for_slices = copy.deepcopy(conv_config) # this is impossible
    conv2d_config_for_slices = conv_config  # todo, this ruins the original config
    # conv2d_config_for_slices.output_layout = ttnn.ROW_MAJOR_LAYOUT, cant set RM, leads to OOM
    conv_output_tensor = None
    conv_prepared_device_weight = None
    conv_prepared_device_bias = None
    conv_output_width = 0

    for input_slice_height_start, input_slice_height_end, pad_top, pad_bottom in conv_height_split_params:
        input_tensor_slice = ttnn.slice(
            input_tensor,
            starts=(0, input_slice_height_start, 0, 0),
            ends=(batch_size, input_slice_height_end, input_width, in_channels),
            steps=(1, 1, 1, 1),
        )

        if pad_top or pad_bottom:
            input_tensor_slice = ttnn.pad(
                input_tensor_slice,
                padding=((0, 0), (pad_top, pad_bottom), (0, 0), (0, 0)),
                value=0,
            )
        # Move to [1, 1, N * H * W, C] format
        input_tensor_slice = input_tensor_slice.reshape(
            1,
            1,
            input_tensor_slice.shape[0] * input_tensor_slice.shape[1] * input_tensor_slice.shape[2],
            input_tensor_slice.shape[3],
        )

        (
            conv_output_slice,
            conv_output_height_slice,
            output_width,
            prepared_device_weight,
            prepared_device_bias,
        ) = ttnn._ttnn.operations.conv.conv2d(
            input_tensor=input_tensor_slice,
            weight_tensor=weight_tensor if conv_prepared_device_weight is None else conv_prepared_device_weight,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=input_slice_height_end - input_slice_height_start + pad_top + pad_bottom,
            input_width=input_width,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, padding[1]),
            dilation=dilation,
            groups=groups,
            bias_tensor=bias_tensor if conv_prepared_device_bias is None else conv_prepared_device_bias,
            conv_config=conv2d_config_for_slices,
            compute_config=compute_config,
            memory_config=memory_config,  # should be ttnn.DRAM_MEMORY_CONFIG, but fails on PCC with that
        )

        # # temp workaround
        # print(f"PRE conv_output_slice.memory_config()={conv_output_slice.memory_config()}")
        # to_layout with memory_config can override memory_config if sharded and on untilize with unpadding path
        conv_output_slice = ttnn.to_layout(conv_output_slice, ttnn.ROW_MAJOR_LAYOUT)
        conv_output_slice = ttnn.to_memory_config(conv_output_slice, ttnn.DRAM_MEMORY_CONFIG)

        if conv_output_tensor is None:
            conv_output_tensor = conv_output_slice
            conv_prepared_device_weight = prepared_device_weight
            conv_prepared_device_bias = prepared_device_bias
            conv_output_width = output_width
        else:
            conv_output_tensor = ttnn.concat([conv_output_tensor, conv_output_slice], dim=2)

    # Restore input tensor to be in [1, 1, N * H * W, C] format
    input_tensor = input_tensor.reshape(
        1, 1, input_tensor.shape[0] * input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )

    if return_output_dim and return_weights_and_bias:
        return (
            conv_output_tensor,
            [conv_output_height, conv_output_width],
            [conv_prepared_device_weight, conv_prepared_device_bias],
        )
    elif return_weights_and_bias:
        return conv_output_tensor, [conv_prepared_device_weight, conv_prepared_device_bias]
    elif return_output_dim:
        return conv_output_tensor, [conv_output_height, conv_output_width]
    else:
        return conv_output_tensor


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
