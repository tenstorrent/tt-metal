# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from typing import Tuple


def golden_maxpool2d(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    **_,
):
    import torch

    input_tensor = input_tensor.reshape(batch_size, input_h, input_w, -1).permute(
        0, 3, 1, 2
    )  # 1, 1, NHW, C -> N, C, H, W

    output_tensor = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )

    N, C, H, W = output_tensor.shape
    output_tensor = output_tensor.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)  # N, C, H, W -> 1, 1, NHW, C

    return output_tensor


ttnn.attach_golden_function(ttnn.max_pool2d, golden_maxpool2d)


def golden_maxpool3d(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_d: int,
    input_h: int,
    input_w: int,
    channels: int,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    **_,
):
    import torch

    input_tensor = input_tensor.reshape(batch_size, input_d, input_h, input_w, -1).permute(
        0, 4, 1, 2, 3
    )  # 1, 1, NDHW, C -> N, C, D, H, W

    output_tensor = torch.nn.functional.max_pool3d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )

    N, C, D, H, W = output_tensor.shape
    output_tensor = output_tensor.permute(0, 2, 3, 4, 1).reshape(
        1, 1, N * D * H * W, C
    )  # N, C, D, H, W -> 1, 1, NDHW, C

    return output_tensor


ttnn.attach_golden_function(ttnn.max_pool3d, golden_maxpool3d)


def golden_avgpool3d(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_d: int,
    input_h: int,
    input_w: int,
    channels: int,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override,
    **_,
):
    import torch

    input_tensor = input_tensor.reshape(batch_size, input_d, input_h, input_w, -1).permute(
        0, 4, 1, 2, 3
    )  # 1, 1, NDHW, C -> N, C, D, H, W

    output_tensor = torch.nn.functional.avg_pool3d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    N, C, D, H, W = output_tensor.shape
    output_tensor = output_tensor.permute(0, 2, 3, 4, 1).reshape(
        1, 1, N * D * H * W, C
    )  # N, C, D, H, W -> 1, 1, NDHW, C

    return output_tensor


ttnn.attach_golden_function(ttnn.avg_pool3d, golden_avgpool3d)


def golden_global_avg_pool2d(input_tensor: ttnn.Tensor):
    import torch

    output_size = (1, 1)
    return torch.nn.functional.global_avg_pool2d(input_tensor, output_size)


ttnn.attach_golden_function(ttnn.global_avg_pool2d, golden_global_avg_pool2d)
