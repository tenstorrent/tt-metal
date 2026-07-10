# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Flash-Attention scaled_dot_product_attention op package."""

from .scaled_dot_product_attention import (
    scaled_dot_product_attention,
    default_compute_kernel_config,
    validate,
    INPUT_TAGGERS,
    SUPPORTED,
    EXCLUSIONS,
)

__all__ = [
    "scaled_dot_product_attention",
    "default_compute_kernel_config",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]
