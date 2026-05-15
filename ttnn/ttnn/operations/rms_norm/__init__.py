# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
rms_norm: Root-mean-square normalization along the last dimension.

    output[..., i, j] = input[..., i, j] / sqrt(mean(input[..., i, :]^2) + epsilon) * gamma[j]
"""

from .rms_norm import rms_norm

__all__ = ["rms_norm"]
