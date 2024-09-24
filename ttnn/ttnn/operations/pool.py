# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def golden_maxpool2d(
    _input_tensor: ttnn.Tensor,
    in_n: int,
    in_h: int,
    in_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    *,
    memory_config: ttnn.MemoryConfig,
    nblocks: int,
    use_multicore: bool,
):
    import torch

    kernel_size = (kernel_h, kernel_w)
    stride = (stride_h, stride_w)
    padding = (pad_h, pad_w)
    dilation = (dilation_h, dilation_w)

    return torch.nn.functional.max_pool2d(
        _input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )


def golden_global_avg_pool2d(input_tensor: ttnn.Tensor):
    import torch

    output_size = (1, 1)
    return torch.nn.functional.global_avg_pool2d(input_tensor, output_size)


ttnn.attach_golden_function(ttnn.global_avg_pool2d, golden_global_avg_pool2d)

avg_pool2d = ttnn.register_python_operation(name="ttnn.avg_pool2d", golden_function=golden_global_avg_pool2d)(
    ttnn._ttnn.operations.pool.avg_pool2d
)
