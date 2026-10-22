# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .matmul_reduce_scatter import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    default_compute_kernel_config,
    matmul_reduce_scatter,
    validate,
)

# Read by tests with getattr and a default; the native op does not fill it.
_LAST_PLAN = {}
