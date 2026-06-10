# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""groupnorm_sc_N_1_HW_C — single-core GroupNorm over (N, 1, H*W, C)."""

from .groupnorm_sc_N_1_HW_C import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    groupnorm_sc_N_1_HW_C,
    validate,
)

__all__ = [
    "groupnorm_sc_N_1_HW_C",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]
