# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Solution: Call the custom eltwise_add operation from Python.

This calls the operation defined in solution_cpp/.
Build: cmake --build build -- onboarding
"""

import torch
import ttnn
from _e03_solution import s03_eltwise_add


def eltwise_add(device, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Call the custom operation (registered via nanobind)
    tt_result = s03_eltwise_add(tt_a, tt_b)

    return ttnn.to_torch(tt_result)
