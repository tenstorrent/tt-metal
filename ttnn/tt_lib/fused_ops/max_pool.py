# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Union, List


def compute_max_pool_shape(kernel_size, stride, padding, x_shape):
    H = x_shape[1]
    W = x_shape[2]
    OH = ((int)((H - kernel_size + 2 * padding) / stride)) + 1
    OW = ((int)((W - kernel_size + 2 * padding) / stride)) + 1
    return [x_shape[0], OH, OW, x_shape[3]]


def run_max_pool_on_device_wrapper(
    device,
    in_n,
    in_h,
    in_w,
    kernel_size,
    stride,
    padding,
    output_mem_config,
    nblocks,
    channels_last=False,
    reshape_2d=False,
):
    def max_pool_2d(x):
        # x_shape_nopad = x.shape_without_padding()
        # out_shape_nopad = compute_max_pool_shape(kernel_size, stride, padding, x_shape_nopad)
        # if reshape_2d and channels_last:
        #     x = x.reshape(x_shape_nopad[0], 1, x_shape_nopad[1] * x_shape_nopad[2], x_shape_nopad[3])
        # out = ttnn.max_pool2d(x, x_shape_nopad[1], x_shape_nopad[2], kernel_size, kernel_size, stride, stride, padding, padding, output_mem_config=output_mem_config, nblocks=nblocks, use_multicore=True)
        out = ttnn.max_pool2d(
            x,
            in_n,
            in_h,
            in_w,
            kernel_size,
            kernel_size,
            stride,
            stride,
            padding,
            padding,
            memory_config=output_mem_config,
            nblocks=nblocks,
            use_multicore=True,
        )
        # if reshape_2d and channels_last:
        #     out = out.reshape(out_shape_nopad[0], out_shape_nopad[1], out_shape_nopad[2], out_shape_nopad[3])
        return out

    return max_pool_2d
