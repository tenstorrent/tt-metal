# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm operation implemented via ttnn.generic_op.

Usage:
    from ttnn.operations.layer_norm import layer_norm
    output = layer_norm(input_tensor, epsilon=1e-5, weight=gamma, bias=beta)
"""

from .layer_norm import layer_norm

__all__ = ["layer_norm"]
