# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .linear_gelu import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    PROPERTIES,
    SUPPORTED,
    default_compute_kernel_config,
    linear_gelu,
    validate,
)

__all__ = [
    "EXCLUSIONS",
    "INPUT_TAGGERS",
    "PROPERTIES",
    "SUPPORTED",
    "default_compute_kernel_config",
    "linear_gelu",
    "validate",
]
