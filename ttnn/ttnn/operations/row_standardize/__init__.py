# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Row Standardize Operation

Performs per-row standardization: (x - mean_row) / sqrt(var_row + epsilon)
Equivalent to LayerNorm without learnable affine parameters.

Usage:
    from ttnn.operations.row_standardize import row_standardize
    output = row_standardize(input_tensor, epsilon=1e-5)
"""

from .row_standardize import row_standardize

__all__ = ["row_standardize"]
