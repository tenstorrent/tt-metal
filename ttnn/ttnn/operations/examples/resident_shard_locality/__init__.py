# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor_with_inline_kernels import (
    LAYOUTS,
    VARIANTS,
    alias_eligible,
    create_program_descriptor,
    resident_shard_copy,
    shard_grid_cores,
    sharded_memory_config,
    tensor_shape,
)

__all__ = [
    "LAYOUTS",
    "VARIANTS",
    "alias_eligible",
    "create_program_descriptor",
    "resident_shard_copy",
    "shard_grid_cores",
    "sharded_memory_config",
    "tensor_shape",
]
