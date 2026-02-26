# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for sharded elementwise add."""

import torch


def sharded_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise add — the reference is simply a + b."""
    return a + b
