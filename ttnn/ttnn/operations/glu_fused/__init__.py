# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
glu_fused — Gated Linear Unit (last-dim split) as a single fused TTNN kernel.

Computes ``torch.nn.functional.glu(x, dim=-1)`` — equivalently
``x[..., :W/2] * sigmoid(x[..., W/2:])`` — in one ``ttnn.generic_op`` dispatch.
The slice/sigmoid/multiply composite is folded into one program: the split
lives at the tile-id level inside the reader; the chain
``Load A → Load B → Sigmoid(B) → Mul(A, B)`` runs entirely inside the SFPU
pipeline on DEST.
"""

from .glu_fused import glu_fused

__all__ = ["glu_fused"]
