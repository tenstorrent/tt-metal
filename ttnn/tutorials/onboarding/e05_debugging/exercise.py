# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exercise: Learn debugging with DPRINT, WATCHER, and tt-triage.

This operation computes sign(input) using 3 kernels: reader -> compute -> writer

WARNING: The exercise has a BUG that causes a HANG!
This is intentional - use tt-triage to diagnose it.

Goals:
1. Run the exercise and observe the hang
2. Use tt-triage to identify which kernel is stuck
3. Add DPRINT statements to understand the issue
4. Find and fix the bug in the compute kernel
"""

import torch
import ttnn
from _e05_exercise import e05_sign


def sign(device: ttnn.Device, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Exercise: Debug the hang, then add DPRINT statements.

    The operation will HANG due to a bug in compute.cpp.
    Use tt-triage to find which kernel is stuck, then fix it.
    """
    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = e05_sign(tt_input)
    return ttnn.to_torch(tt_output)
