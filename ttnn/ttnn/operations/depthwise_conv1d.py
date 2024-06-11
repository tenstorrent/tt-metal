# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict, Optional
import torch
import warnings
import math
import ttnn

import ttnn.experimental

DepthwiseConv1dConfig = ttnn._ttnn.operations.depthwise_conv1d.DepthwiseConv1dConfig


def _depthwise_conv_op_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(name="ttnn.depthwise_conv1d", validate_input_tensors=_depthwise_conv_op_validate_input_tensors)
def depthwise_conv1d(
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
    conv_config: DepthwiseConv1dConfig = None,  # config overrides by user
    conv_op_cache={},  # basic conv object caching in python needed for intermediate refactoring. Not needed after full op refactoring in C++.
    debug=False,
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
    run_new_conv = True
    if debug:
        deallocate_act_debug_mode = conv_config.deallocate_activation
        conv_config.deallocate_activation = False
    if run_new_conv:
        (
            output_tensor_new,
            output_height_new,
            output_width_new,
            weight_tensor_on_dev_new,
            bias_tensor_on_dev_new,
        ) = ttnn._ttnn.operations.depthwise_conv1d.depthwise_conv1d(
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
        )
        if not debug:
            return (
                output_tensor_new,
                output_height_new,
                output_width_new,
                weight_tensor_on_dev_new,
                bias_tensor_on_dev_new,
            )


__all__ = []
