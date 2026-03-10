# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Group Norm Operation

Implements Group Normalization using ttnn.generic_op with ProgramDescriptor APIs.

Usage:
    from ttnn.operations.group_norm import group_norm
    output = group_norm(input_tensor, num_groups=G, gamma=gamma, beta=beta)
"""

from .group_norm import group_norm

__all__ = ["group_norm"]
