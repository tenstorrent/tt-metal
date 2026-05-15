# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
rms_norm: Root-mean-square normalization along the last dimension.

    output[..., i, j] = input[..., i, j] / sqrt(mean(input[..., i, :]^2) + epsilon) * gamma[j]

Registry exports: SHAPE_TAGGERS, SUPPORTED, EXCLUSIONS, validate are
imported by eval/golden_tests/rms_norm/test_golden.py.
"""

from .rms_norm import (
    EXCLUSIONS,
    SHAPE_TAGGERS,
    SUPPORTED,
    rms_norm,
    validate,
)

__all__ = [
    "EXCLUSIONS",
    "SHAPE_TAGGERS",
    "SUPPORTED",
    "rms_norm",
    "validate",
]
