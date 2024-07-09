# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict


import sys
import ttnn


__all__ = []


def golden_global_avg_pool2d(input_tensor: ttnn.Tensor):
    import torch

    output_size = (1, 1)
    return torch.nn.functional.global_avg_pool2d(input_tensor, output_size)


global_avg_pool2d = ttnn.register_python_operation(
    name="ttnn.global_avg_pool2d", golden_function=golden_global_avg_pool2d
)(ttnn._ttnn.operations.avgpool.global_avg_pool2d)

avg_pool2d = ttnn.register_python_operation(name="ttnn.avg_pool2d", golden_function=golden_global_avg_pool2d)(
    ttnn._ttnn.operations.avgpool.avg_pool2d
)


__all__ = []
