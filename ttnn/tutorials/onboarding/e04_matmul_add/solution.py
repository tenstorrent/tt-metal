# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Solution: Call the custom matmul_add operation from Python.

This calls the operation defined in solution_cpp/.
Build: cmake --build build -- onboarding
"""

import torch
import ttnn
from _e04_solution import s04_matmul_add


def matmul_add(device, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_c = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Call the custom operation (registered via nanobind)
    tt_result = s04_matmul_add(tt_a, tt_b, tt_c)

    return ttnn.to_torch(tt_result)
