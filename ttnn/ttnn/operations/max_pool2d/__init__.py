# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .max_pool2d import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    default_compute_kernel_config,
    max_pool2d,
    validate,
)

__all__ = [
    "max_pool2d",
    "SUPPORTED",
    "EXCLUSIONS",
    "INPUT_TAGGERS",
    "validate",
    "default_compute_kernel_config",
]
