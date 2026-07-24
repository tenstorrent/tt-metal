# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated perf bake-off for rms_norm pass-1 (Sigma x^2 reduce) restructure.

Author: blocking-perf-part-optimizer. NOT the real op — a self-contained micro-bench
reconstructing ONLY the per-core pass-1 compute of rms_norm's cross-core kernel
(kernels/rms_norm_xcore_compute.cpp, the do_pass1 lambda). Uncommitted artifact.
"""

from .program_descriptor_with_inline_kernels import (
    BASELINE,
    VARIANTS,
    create_sharded_memory_config,
    input_shape,
    run_op,
)

__all__ = [
    "BASELINE",
    "VARIANTS",
    "create_sharded_memory_config",
    "input_shape",
    "run_op",
]
