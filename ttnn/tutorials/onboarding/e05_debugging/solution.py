# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
E05 Solution: Working sign implementation.

Computes element-wise sign: -1 if input < 0, 0 if input == 0, 1 if input > 0
Build: cmake --build build -- onboarding
"""

import torch
import ttnn
from _e05_solution import s05_sign


def sign(device: ttnn.Device, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise sign of input tensor.

    Args:
        device: TT device
        input_tensor: Input tensor

    Returns:
        Sign of input tensor (-1, 0, or 1)
    """
    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = s05_sign(tt_input)

    return ttnn.to_torch(tt_output)
