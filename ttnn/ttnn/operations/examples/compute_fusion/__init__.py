# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor_with_inline_kernels import (
    PHASE_ZONES,
    SCENARIOS,
    VARIANTS,
    create_program_descriptor,
    create_sharded_memory_config,
    run_fusion,
    variants_for,
)

__all__ = [
    "PHASE_ZONES",
    "SCENARIOS",
    "VARIANTS",
    "create_program_descriptor",
    "create_sharded_memory_config",
    "run_fusion",
    "variants_for",
]
