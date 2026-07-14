# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm operation (registry model)."""

from .rms_norm import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    PROPERTIES,
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
    "PROPERTIES",
]
