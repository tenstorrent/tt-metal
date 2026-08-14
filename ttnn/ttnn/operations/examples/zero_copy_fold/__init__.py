# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor_with_inline_kernels import (
    BASELINE,
    VARIANTS,
    create_program_descriptor,
    run_op,
    sharded_memory_config,
)

__all__ = [
    "BASELINE",
    "VARIANTS",
    "create_program_descriptor",
    "run_op",
    "sharded_memory_config",
]
