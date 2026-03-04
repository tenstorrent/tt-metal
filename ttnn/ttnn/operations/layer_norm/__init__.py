# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm Operation

Implements y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
over the last dimension of a 2D row-major input tensor.

Usage:
    from ttnn.operations.layer_norm import layer_norm
    output = layer_norm(input_tensor)
    output = layer_norm(input_tensor, weight=gamma, bias=beta, eps=1e-5)
"""

from .layer_norm import layer_norm

__all__ = ["layer_norm"]
