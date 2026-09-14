# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — RMSNorm over the last dimension, one native device program per call.

    from ttnn.operations.rms_norm import rms_norm
"""

from .rms_norm import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    default_compute_kernel_config,
    rms_norm,
    validate,
)

__all__ = [
    "rms_norm",
    "validate",
    "default_compute_kernel_config",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]
