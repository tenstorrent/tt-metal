# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.operations.softmax

Numerically-stable row-wise (dim=-1) or column-wise (dim=-2) softmax for
fp32 TILE-layout 4D tensors.
"""

from .softmax import softmax

__all__ = ["softmax"]
