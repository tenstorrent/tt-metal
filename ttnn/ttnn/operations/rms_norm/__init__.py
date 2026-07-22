# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm op package (registry model).

Re-exports the public entry point plus the four registry declarations the
golden harness and axis-tagger import from `ttnn.operations.rms_norm`.
"""

from .rms_norm import (
    rms_norm,
    validate,
    default_compute_kernel_config,
    INPUT_TAGGERS,
    SUPPORTED,
    EXCLUSIONS,
)

__all__ = [
    "rms_norm",
    "validate",
    "default_compute_kernel_config",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]
