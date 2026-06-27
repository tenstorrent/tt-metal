# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""2D dual-multicast matmul op (registry model)."""

from .matmul import (
    matmul,
    validate,
    default_compute_kernel_config,
    INPUT_TAGGERS,
    SUPPORTED,
    EXCLUSIONS,
)

__all__ = [
    "matmul",
    "validate",
    "default_compute_kernel_config",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]
