# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .avg_pool2d import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    avg_pool2d,
    default_compute_kernel_config,
    validate,
)

__all__ = ["avg_pool2d", "SUPPORTED", "EXCLUSIONS", "INPUT_TAGGERS", "validate", "default_compute_kernel_config"]
