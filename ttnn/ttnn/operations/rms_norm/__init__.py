# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm Operation

RMS (Root Mean Square) normalization along the last dimension.

Usage:
    from ttnn.operations.rms_norm import rms_norm
    output = rms_norm(input_tensor, gamma=gamma_tensor, epsilon=1e-6)
"""

from .rms_norm import rms_norm

__all__ = ["rms_norm"]
