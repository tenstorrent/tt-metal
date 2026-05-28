# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.operations.layer_norm

Thin alias module for the immutable acceptance test
(`tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm.py`)
which imports `from ttnn.operations.layer_norm import layer_norm`. The
real implementation lives at `ttnn.operations.layer_norm_rm`.
"""

from ttnn.operations.layer_norm_rm import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    layer_norm_rm as layer_norm,
    validate,
)

__all__ = [
    "layer_norm",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
    "validate",
]
