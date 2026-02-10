# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM Operation

Layer normalization for row-major tensors. Normalizes each row independently
using mean and variance, then applies learned affine transformation.

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor, gamma, beta, epsilon=1e-5)
"""

from .layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
