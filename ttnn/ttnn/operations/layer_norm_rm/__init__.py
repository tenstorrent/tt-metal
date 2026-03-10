# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Row-wise Layer Normalization on ROW_MAJOR Interleaved Tensors

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma, beta, epsilon=1e-6)
"""

from .layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
