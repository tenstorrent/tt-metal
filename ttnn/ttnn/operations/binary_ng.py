# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


ttnn.attach_golden_function(ttnn.experimental.add, golden_function=lambda a, b: a + b)
ttnn.attach_golden_function(ttnn.experimental.sub, golden_function=lambda a, b: a - b)
ttnn.attach_golden_function(ttnn.experimental.rsub, golden_function=lambda a, b: b - a)
ttnn.attach_golden_function(ttnn.experimental.mul, golden_function=lambda a, b: a * b)
ttnn.attach_golden_function(ttnn.experimental.div, golden_function=lambda a, b: torch.divide(a, b))
ttnn.attach_golden_function(ttnn.experimental.eq, golden_function=lambda a, b: torch.eq(a, b))
ttnn.attach_golden_function(ttnn.experimental.ne, golden_function=lambda a, b: torch.ne(a, b))
ttnn.attach_golden_function(ttnn.experimental.gt, golden_function=lambda a, b: torch.gt(a, b))
ttnn.attach_golden_function(ttnn.experimental.lt, golden_function=lambda a, b: torch.lt(a, b))
ttnn.attach_golden_function(ttnn.experimental.gte, golden_function=lambda a, b: torch.ge(a, b))
ttnn.attach_golden_function(ttnn.experimental.lte, golden_function=lambda a, b: torch.le(a, b))
ttnn.attach_golden_function(ttnn.experimental.ldexp, golden_function=lambda a, b: torch.ldexp(a, b))
ttnn.attach_golden_function(ttnn.experimental.logaddexp, golden_function=lambda a, b: torch.logaddexp(a, b))
ttnn.attach_golden_function(ttnn.experimental.logaddexp2, golden_function=lambda a, b: torch.logaddexp2(a, b))
ttnn.attach_golden_function(ttnn.experimental.logical_and, golden_function=lambda a, b: torch.logical_and(a, b))
ttnn.attach_golden_function(ttnn.experimental.logical_or, golden_function=lambda a, b: torch.logical_or(a, b))
ttnn.attach_golden_function(ttnn.experimental.logical_xor, golden_function=lambda a, b: torch.logical_xor(a, b))
ttnn.attach_golden_function(
    ttnn.experimental.squared_difference, golden_function=lambda a, b: torch.square(torch.sub(a, b))
)
ttnn.attach_golden_function(
    ttnn.experimental.bias_gelu, golden_function=lambda a, b: torch.nn.functional.gelu(torch.add(a, b))
)
