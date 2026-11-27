# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from .g1_minimal_matmul_split import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    default_compute_kernel_config,
    g1_minimal_matmul_split,
    validate,
)

__all__ = [
    "g1_minimal_matmul_split",
    "default_compute_kernel_config",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
    "validate",
]
