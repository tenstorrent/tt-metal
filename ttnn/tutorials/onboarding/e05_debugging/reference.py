# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
E05 Reference: PyTorch reference implementation for sign operation.
"""

import torch


def sign(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation: sign function.
    output = sign(input) = -1 if input < 0, 0 if input == 0, 1 if input > 0

    Args:
        input_tensor: Input tensor

    Returns:
        Sign of input tensor
    """
    return torch.sign(input_tensor)
