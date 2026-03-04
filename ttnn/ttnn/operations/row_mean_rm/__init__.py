# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
row_mean_rm - Row-wise mean on row-major interleaved tensors.

Import as:
    from ttnn.operations.row_mean_rm import row_mean_rm
"""

from .row_mean_rm import row_mean_rm

__all__ = ["row_mean_rm"]
