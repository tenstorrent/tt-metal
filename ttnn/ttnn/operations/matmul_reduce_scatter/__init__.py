# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .matmul_reduce_scatter import (
    default_compute_kernel_config,
    matmul_reduce_scatter,
    matmul_reduce_scatter_weight_prefetch,
)
from .matmul_reduce_scatter_program_descriptor import WeightPrefetch, _LAST_PLAN  # _LAST_PLAN: the last call's plan
