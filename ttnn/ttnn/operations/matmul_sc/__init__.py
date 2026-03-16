# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
matmul_sc: Single-core tiled matrix multiplication C = A x B.

Usage:
    from ttnn.operations.matmul_sc import matmul_sc
    output = matmul_sc(input_a, input_b)
"""

from .matmul_sc import matmul_sc

__all__ = ["matmul_sc"]
