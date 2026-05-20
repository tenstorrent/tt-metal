# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shim module that re-exports layer_norm from the layer_norm_rm package, so callers
can do `from ttnn.operations.layer_norm import layer_norm`.
"""

from ttnn.operations.layer_norm_rm import layer_norm

__all__ = ["layer_norm"]
