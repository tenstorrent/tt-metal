# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Optional


import tt_lib as ttl

import ttnn


def _create_golden_function(torch_function_name):
    import torch

    torch_function = getattr(torch, torch_function_name)

    def golden_function(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]] = None, keepdim=False, **_):
        if dim == None:
            return torch_function(input_tensor, keepdim=keepdim)
        else:
            return torch_function(input_tensor, dim=dim, keepdim=keepdim)

    return golden_function


# Generic reductions
mean = ttnn.register_operation(golden_function=_create_golden_function("mean"))(ttnn._ttnn.operations.reduction.mean)
sum = ttnn.register_operation(golden_function=_create_golden_function("sum"))(ttnn._ttnn.operations.reduction.sum)
max = ttnn.register_operation(golden_function=_create_golden_function("max"))(ttnn._ttnn.operations.reduction.max)
min = ttnn.register_operation(golden_function=_create_golden_function("min"))(ttnn._ttnn.operations.reduction.min)
var = ttnn.register_operation(golden_function=_create_golden_function("var"))(ttnn._ttnn.operations.reduction.var)
std = ttnn.register_operation(golden_function=_create_golden_function("std"))(ttnn._ttnn.operations.reduction.std)

# Special reductions
argmax = ttnn.register_operation(golden_function=_create_golden_function("argmax"))(
    ttnn._ttnn.operations.reduction.argmax
)


__all__ = []
