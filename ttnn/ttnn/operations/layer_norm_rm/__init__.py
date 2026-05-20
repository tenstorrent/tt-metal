# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — LayerNorm operation supporting both ROW_MAJOR and TILE inputs.

Import as:
    from ttnn.operations.layer_norm_rm import layer_norm
"""

from .layer_norm_rm import layer_norm

__all__ = ["layer_norm"]
