# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm Operation

y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

Import as:
    from ttnn.operations.layernorm import layernorm
"""

from .layernorm import layernorm

__all__ = ["layernorm"]
