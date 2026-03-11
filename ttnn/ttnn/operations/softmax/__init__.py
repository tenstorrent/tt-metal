# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax Operation

Computes numerically-stable softmax along a specified dimension.

Usage:
    from ttnn.operations.softmax import softmax
    output = softmax(input_tensor, dim=-1)
"""

from .softmax import softmax

__all__ = ["softmax"]
