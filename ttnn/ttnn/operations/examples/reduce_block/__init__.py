# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor_with_inline_kernels import (
    BASELINE,
    DIMS,
    DTYPES,
    VARIANTS,
    create_accumulate_program_descriptor,
    create_program_descriptor,
    create_sharded_memory_config,
    dispatch_min,
    elements_reduced,
    input_shape,
    out_tile_count,
    output_shape,
    reduced_count,
    run_accumulate,
    run_op,
)

__all__ = [
    "BASELINE",
    "DIMS",
    "DTYPES",
    "VARIANTS",
    "create_accumulate_program_descriptor",
    "create_program_descriptor",
    "create_sharded_memory_config",
    "dispatch_min",
    "elements_reduced",
    "input_shape",
    "out_tile_count",
    "output_shape",
    "reduced_count",
    "run_accumulate",
    "run_op",
]
