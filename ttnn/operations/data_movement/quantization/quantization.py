# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

def quantize(input_tensor, scale, zero_point, output_dtype=ttnn.uint8):
    # Formula: clamp(round(input / scale + zero_point), 0, 255)
    output = ttnn.add(ttnn.multiply(input_tensor, 1.0 / scale), zero_point)
    output = ttnn.round(output)
    if output_dtype == ttnn.uint8:
        output = ttnn.clamp(output, 0, 255)
    return ttnn.typecast(output, output_dtype)

def requantize(input_tensor, input_scale, input_zero_point, output_scale, output_zero_point, output_dtype=ttnn.uint8):
    # Formula: clamp(round((input - input_zero_point) * input_scale / output_scale + output_zero_point), 0, 255)
    output = ttnn.subtract(input_tensor, input_zero_point)
    output = ttnn.multiply(output, input_scale / output_scale)
    output = ttnn.add(output, output_zero_point)
    output = ttnn.round(output)
    if output_dtype == ttnn.uint8:
        output = ttnn.clamp(output, 0, 255)
    return ttnn.typecast(output, output_dtype)