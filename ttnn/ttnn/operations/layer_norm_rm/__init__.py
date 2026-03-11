# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Row-major layer normalization via generic_op.

Normalizes over the last dimension (W) of a row-major interleaved tensor.
Tilize/untilize happen in-kernel (no host-side layout conversion).

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor, gamma=gamma, beta=beta, epsilon=1e-5)
"""

from .layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
