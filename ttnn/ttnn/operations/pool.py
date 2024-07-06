# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict

import sys
import ttnn

from ttnn.operations.conv.tt_py_max_pool import (
    TTPyMaxPool,
    SlidingWindowOpParams,
    SlidingWindowOpParamsWithParallelConfig,
)

import tt_lib as ttl

__all__ = []


class MaxPool2d:
    r"""
    Applies a 2D max pooling over an input signal composed of several input planes.

    If `padding` is non-zero, then the input is implicitly padded with negative infinity on both sides for padding number of points.
    `dilation` controls the spacing between the kernel points.

    Arguments:
        * :attr: kernel_size (Union[int, Tuple[int, int]]): the size of the window to take a max over
        * :attr: stride (Union[int, Tuple[int, int]]): the stride of the window. Default value is 1
        * :attr: padding (Union[int, Tuple[int, int]]): Implicit negative infinity padding to be added on both sides
        * :attr: dilation (Union[int, Tuple[int, int]]): a parameter that controls the stride of window elements
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        dtype: ttnn.DataType = None,
        *,
        device: ttnn.Device,
        batch_size: int,
        input_height: int,
        input_width: int,
        reader_patterns_cache: Dict,
        parallel_config_override: Dict = None,
        deallocate_activation: bool = False,
        channels: int = None,
    ):
        if isinstance(kernel_size, int):
            window_h = kernel_size
            window_w = kernel_size
        else:
            window_h, window_w = kernel_size

        if isinstance(stride, int):
            stride_h = stride
            stride_w = stride
        else:
            stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h = padding
            pad_w = padding
        else:
            pad_h, pad_w = padding

        if isinstance(dilation, int):
            dilation_h = dilation
            dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        assert dilation_h == 1, f"Only dilation_h = 1 supported. Found dilation_h={dilation_h}"
        assert dilation_w == 1, f"Only dilation_w = 1 supported. Found dilation_w={dilation_w}"

        sliding_window_op_params = SlidingWindowOpParams(
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            window_h=window_h,
            window_w=window_w,
            batch_size=batch_size,
            input_h=input_height,
            input_w=input_width,
        )
        self.max_pool = TTPyMaxPool(
            sliding_window_op_params,
            device,
            reader_patterns_cache,
            pad_val=0xF7FF,
            parallel_config_override=parallel_config_override,
            deallocate_activation=deallocate_activation,
            act_dtype=dtype,
            channels=channels,
            pool_op=max_pool2d_v2,
        )

    @ttnn.register_python_operation(name="ttnn.MaxPool2d.__call__", is_method=True)
    def __call__(self, activation: ttnn.Tensor):
        return self.max_pool(activation)

    @ttnn.register_python_operation(name="ttnn.MaxPool2d.copy_input_to_device", is_method=True)
    def copy_input_to_device(self, input: ttnn.Tensor):
        return self.max_pool.copy_input_to_device(input)

    @ttnn.register_python_operation(
        name="ttnn.MaxPool2d.copy_output_from_device",
        is_method=True,
    )
    def copy_output_from_device(self, output: ttnn.Tensor):
        return self.max_pool.copy_output_from_device(output)


## Average Pooling


def golden_global_avg_pool2d(input_tensor: ttnn.Tensor):
    import torch

    output_size = (1, 1)
    return torch.nn.functional.global_avg_pool2d(input_tensor, output_size)


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


global_avg_pool2d = ttnn._ttnn.operations.avgpool.global_avg_pool2d
avg_pool2d = ttnn._ttnn.operations.avgpool.avg_pool2d
max_pool2d = ttnn.ttnn._ttnn.operations.maxpool.max_pool2d
max_pool2d_v2 = ttnn._ttnn.operations.maxpool.max_pool2d_v2

ttnn.attach_golden_function(global_avg_pool2d, golden_function=golden_global_avg_pool2d)
# ttnn.attach_golden_function(avg_pool2d, golden_function=golden_global_avg_pool2d)

# ttnn.attach_golden_function(max_pool2d_v2, golden_function=golden_maxpool2d)
# ttnn.attach_golden_function(max_pool2d_v2, golden_function=golden_maxpool2d)


__all__ = []
