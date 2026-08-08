# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measured auto-configuration helpers exposed as ``ttnn.experimental.auto_config``."""

from ._selector import WeightPlacement, explain_matmul, place_weight
from .api import linear, matmul

__all__ = [
    "WeightPlacement",
    "explain_matmul",
    "linear",
    "matmul",
    "place_weight",
]
