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

# Test hooks (read / written at call time through this package module, so monkeypatch.setattr works):
#   _FORCE_XBLOCK_RESIDENCY: None (choose_k_block decides) | True | False
#   _LAST_PLAN: overwritten on every call with the regime / schedule / grid the call used.
_FORCE_XBLOCK_RESIDENCY = None
_LAST_PLAN = {}
