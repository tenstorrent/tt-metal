# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.operations.linear — generic_op-based linear (matmul + optional bias)
implementation that exercises the kernel_lib matmul_block + bias_add helpers
end-to-end.
"""

from .linear import linear

__all__ = ["linear"]
