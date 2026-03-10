# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Layer Normalization on Row-Major Interleaved Tensors

Implements layer normalization across the width (last) dimension.
Input and output are ROW_MAJOR layout, bfloat16, interleaved in DRAM.

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm

    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma)
    output = layer_norm_rm(input_tensor, gamma, beta)
    output = layer_norm_rm(input_tensor, gamma, beta, epsilon=1e-5)
"""

from .layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
