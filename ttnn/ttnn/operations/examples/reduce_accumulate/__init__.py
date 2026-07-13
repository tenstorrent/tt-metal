# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor_with_inline_kernels import (
    BASELINE,
    DIMS,
    DTYPES,
    VARIANTS,
    create_program_descriptor,
    create_sharded_memory_config,
    dispatch_min,
    elements_reduced,
    input_shape,
    run_op,
)

__all__ = [
    "BASELINE",
    "DIMS",
    "DTYPES",
    "VARIANTS",
    "create_program_descriptor",
    "create_sharded_memory_config",
    "dispatch_min",
    "elements_reduced",
    "input_shape",
    "run_op",
]
