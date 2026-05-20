# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shim module that re-exports the layer_norm op from the layer_norm_rm package,
along with the four registry-model symbols (INPUT_TAGGERS, SUPPORTED,
EXCLUSIONS, validate) so the golden test suite (eval/golden_tests/layer_norm_rm)
can do `from ttnn.operations.layer_norm import ...`.
"""

from ttnn.operations.layer_norm_rm import layer_norm
from ttnn.operations.layer_norm_rm.layer_norm_rm import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    validate,
)

__all__ = ["layer_norm", "validate", "INPUT_TAGGERS", "SUPPORTED", "EXCLUSIONS"]
