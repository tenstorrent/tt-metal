# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for eltwise add."""

import torch


def eltwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition: a + b"""
    return a + b
