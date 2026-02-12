# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exercise: Call your custom matmul_add operation from Python.

After implementing exercise_cpp/, this should work:
    result = ttnn.onboarding_matmul_add(tt_a, tt_b, tt_c)
"""

import torch
import ttnn


def matmul_add(device, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_c = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # TODO: Call your custom operation after implementing exercise_cpp/
    # tt_result = ttnn.onboarding_matmul_add(tt_a, tt_b, tt_c)
    # return ttnn.to_torch(tt_result)

    raise NotImplementedError("Implement exercise_cpp/ first")
