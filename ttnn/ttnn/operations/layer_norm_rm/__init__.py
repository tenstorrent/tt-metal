# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Layer Normalization on Row-Major Interleaved Tensors

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor, gamma=gamma_tensor, beta=beta_tensor)
"""

from .layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
