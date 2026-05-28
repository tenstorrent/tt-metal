# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — row-wise layer normalization for ROW_MAJOR_LAYOUT float32 tensors.

The kernels accept RM input directly (in-kernel tilize) and produce RM
output (in-kernel untilize); no host-side layout conversion required.

Registry-model artefacts (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate)
are re-exported here so the golden-test suite can import them directly
from ``ttnn.operations.layer_norm_rm``.
"""

from .layer_norm_rm import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    layer_norm,
    validate,
)

__all__ = [
    "EXCLUSIONS",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "layer_norm",
    "validate",
]
