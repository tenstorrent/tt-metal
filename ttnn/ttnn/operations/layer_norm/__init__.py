# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm operation for TT-NN.

Usage:
    from ttnn.operations.layer_norm import layer_norm

    output = layer_norm(input_tensor)
    output = layer_norm(input_tensor, gamma=gamma_tt, beta=beta_tt, eps=1e-5)
"""

from .layer_norm import layer_norm

__all__ = ["layer_norm"]
