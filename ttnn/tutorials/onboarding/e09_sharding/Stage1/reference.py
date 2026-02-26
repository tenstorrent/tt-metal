# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for interleaved to sharded (identity operation)."""

import torch


def interleaved_to_sharded(input: torch.Tensor) -> torch.Tensor:
    """Identity — the data doesn't change, only the memory config does."""
    return input.clone()
