# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor_with_inline_kernels import (
    BASELINE,
    VARIANTS,
    block_rows_for,
    create_program_descriptor,
    create_sharded_memory_config,
    run_op,
    variant_is_valid,
)

__all__ = [
    "BASELINE",
    "VARIANTS",
    "block_rows_for",
    "create_program_descriptor",
    "create_sharded_memory_config",
    "run_op",
    "variant_is_valid",
]
