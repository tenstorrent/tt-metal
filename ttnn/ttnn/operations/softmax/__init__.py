# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.operations.softmax

Numerically-stable row-wise (dim=-1) or column-wise (dim=-2) softmax for
fp32 TILE-layout 4D tensors.

Registry-model artefacts (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate)
are re-exported here so the golden-test suite can import them directly
from `ttnn.operations.softmax`.
"""

from .softmax import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    softmax,
    validate,
)

__all__ = [
    "EXCLUSIONS",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "softmax",
    "validate",
]
