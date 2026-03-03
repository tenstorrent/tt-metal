# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Row-Major Layer Normalization

Computes: output = gamma * (x - mean) / sqrt(var + eps) + beta
for each row of the input tensor (layer normalization along the W dimension).

Input tensor must be in ROW_MAJOR layout, bfloat16, with last two dims tile-aligned.
"""

from .layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
