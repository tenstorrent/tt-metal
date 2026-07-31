# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor_with_inline_kernels import (
    BASELINE,
    CB_DEPTH_BLOCKS,
    LABEL,
    NOCS,
    RISCS,
    VARIANTS,
    compute_chunk,
    create_output_memory_config,
    create_program_descriptor,
    dual_noc_read,
    l1_footprint_bytes,
    validate,
)

__all__ = [
    "BASELINE",
    "CB_DEPTH_BLOCKS",
    "LABEL",
    "NOCS",
    "RISCS",
    "VARIANTS",
    "compute_chunk",
    "create_output_memory_config",
    "create_program_descriptor",
    "dual_noc_read",
    "l1_footprint_bytes",
    "validate",
]
