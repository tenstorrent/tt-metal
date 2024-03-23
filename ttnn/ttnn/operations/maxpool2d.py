# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict

import tt_lib as ttl

import ttnn

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_max_pool import (
    TTPyMaxPool,
    SlidingWindowOpParams,
)


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
        )

    @ttnn.register_operation(
        name="ttnn.MaxPool2d.__call__", validate_input_tensors=lambda *args, **kwargs: None, is_method=True
    )
    def __call__(self, activation: ttnn.Tensor):
        return self.max_pool(activation)

    @ttnn.register_operation(
        name="ttnn.MaxPool2d.copy_input_to_device", validate_input_tensors=lambda *args, **kwargs: None, is_method=True
    )
    def copy_input_to_device(self, input: ttnn.Tensor):
        return self.max_pool.copy_input_to_device(input)

    @ttnn.register_operation(
        name="ttnn.MaxPool2d.copy_output_from_device",
        validate_input_tensors=lambda *args, **kwargs: None,
        is_method=True,
    )
    def copy_output_from_device(self, output: ttnn.Tensor):
        return self.max_pool.copy_output_from_device(output)


## Average Pooling


def _torch_global_avg_pool2d(input_tensor: ttnn.Tensor):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    output_size = (1, 1)
    return torch.nn.functional.global_avg_pool2d(input_tensor, output_size)


def _global_avg_pool2d_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.global_avg_pool2d",
    validate_input_tensors=_global_avg_pool2d_validate_input_tensors,
    torch_function=_torch_global_avg_pool2d,
)
def global_avg_pool2d(input_tensor: ttnn.Tensor, out_memory_config: ttnn.MemoryConfig = None) -> ttnn.Tensor:
    r"""
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    Arguments:
        * :attr: input_tensor: the input tensor
    """
    if out_memory_config is None:
        output = ttl.tensor.average_pool_2d(input_tensor)
    else:
        output = ttl.tensor.average_pool_2d(input_tensor, out_memory_config)
    return output
