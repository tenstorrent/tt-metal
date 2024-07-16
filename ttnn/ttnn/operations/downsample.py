# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple


import ttnn
import torch


def _golden_function(
    input_tensor: ttnn.Tensor,
    downsample_params: Tuple[float, float, float, float, float],
    output_dtype: ttnn.DataType,
    **_,
):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    ret = torch.nn.functional.downsample(input_tensor, downsample_params=downsample_params, dtype=dtype)
    ret = ret.permute(0, 2, 3, 1)
    return ret


downsample = ttnn.attach_golden_function(ttnn.downsample, golden_function=_golden_function)

__all__ = []
