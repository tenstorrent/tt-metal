# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

from .matmul_reduce_scatter import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    default_compute_kernel_config,
    matmul_reduce_scatter,
    validate,
)

# _LAST_PLAN: read through this package module, the regime / schedule / grid the last call of the C++ op used.


class _Package(types.ModuleType):
    def __getattr__(self, name):
        if name == "_LAST_PLAN":
            import ttnn

            return ttnn._ttnn.operations.matmul_reduce_scatter._last_plan()
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _Package
