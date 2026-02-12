# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for matmul + add."""

import torch


def matmul_add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute a @ b + c"""
    return torch.matmul(a, b) + c
