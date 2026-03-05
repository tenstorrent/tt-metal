# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Layer normalization over the last dimension of a row-major tensor.

Import as:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
"""

from .layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
