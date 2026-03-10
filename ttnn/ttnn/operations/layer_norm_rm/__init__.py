# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Row-major layer normalization via generic_op.

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor, gamma=gamma, beta=beta, epsilon=1e-5)
"""

from .layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
