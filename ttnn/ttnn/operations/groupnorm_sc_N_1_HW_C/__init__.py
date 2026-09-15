# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""groupnorm_sc_N_1_HW_C — GroupNorm over a channel-last (N, 1, H*W, C) tensor."""

from .groupnorm_sc_N_1_HW_C import (
    groupnorm_sc_N_1_HW_C,
    default_compute_kernel_config,
    set_l1_budget_bytes_override,
    set_max_cores_override,
    validate,
    INPUT_TAGGERS,
    SUPPORTED,
    EXCLUSIONS,
)

__all__ = [
    "groupnorm_sc_N_1_HW_C",
    "default_compute_kernel_config",
    "set_l1_budget_bytes_override",
    "set_max_cores_override",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]
