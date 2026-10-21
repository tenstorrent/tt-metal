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

# Kept for tests that read it (the native op does not report its plan, so it stays empty).
_LAST_PLAN = {}
