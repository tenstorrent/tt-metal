# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — row-wise layer normalization for ROW_MAJOR_LAYOUT float32 tensors.

The kernels accept RM input directly (in-kernel tilize) and produce RM output
(in-kernel untilize); no host-side layout conversion required.
"""

from .layer_norm_rm import layer_norm

__all__ = ["layer_norm"]
