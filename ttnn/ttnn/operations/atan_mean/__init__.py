# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
atan_mean — fused atan-then-row-mean along the last dim.

Computes torch.atan(x).mean(dim=-1) as a single fused TTNN kernel: SFPU atan is
applied per-tile and the row mean is reduced inside the same program, so the
intermediate atan(x) tensor is never materialised to DRAM.

Usage:
    from ttnn.operations.atan_mean import atan_mean
    output = atan_mean(input_tensor)
"""

from .atan_mean import atan_mean

__all__ = ["atan_mean"]
