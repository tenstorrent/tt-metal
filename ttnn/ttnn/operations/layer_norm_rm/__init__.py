# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn.operations.layer_norm_rm

Per-row (final-dim) LayerNorm for ROW_MAJOR fp32 tensors. See
op_design.md for the full Phase-0 specification.
"""

from .layer_norm_rm import layer_norm_rm

# Aliased name for the immutable acceptance test, which imports the public
# entry as `layer_norm`.
layer_norm = layer_norm_rm

__all__ = ["layer_norm_rm", "layer_norm"]
