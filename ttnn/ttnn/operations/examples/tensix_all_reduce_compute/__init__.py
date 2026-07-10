# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor_with_inline_kernels import (
    VARIANTS,
    create_program_descriptor,
    create_sharded_memory_config,
    reduce_blocks,
)

__all__ = ["VARIANTS", "create_program_descriptor", "create_sharded_memory_config", "reduce_blocks"]
