# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict, Optional
import torch
import warnings
import math
import ttnn

Conv1dConfig = ttnn._ttnn.operations.conv2d.Conv2dConfig


@ttnn.register_python_operation(name="ttnn.Conv1d")
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
    conv_op_cache={},  # basic conv object caching in python needed for intermediate refactoring. Not needed after full op refactoring in C++.
    debug=False,
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
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
    ) = ttnn._ttnn.operations.conv2d.conv2d(
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
    )

    return (
        output_tensor_new,
        output_length_new,
        weight_tensor_on_dev_new,
        bias_tensor_on_dev_new,
    )


__all__ = []
