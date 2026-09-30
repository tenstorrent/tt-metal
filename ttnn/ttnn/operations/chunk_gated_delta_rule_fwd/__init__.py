# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""chunk_gated_delta_rule_fwd — forward of the chunked gated delta rule (Gated DeltaNet)."""

from .chunk_gated_delta_rule_fwd import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    PROPERTIES,
    SUPPORTED,
    chunk_gated_delta_rule_fwd,
    default_compute_kernel_config,
    validate,
)

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "default_compute_kernel_config",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
    "PROPERTIES",
]
