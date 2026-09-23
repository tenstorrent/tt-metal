# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""groupnorm_sc_N_1_HW_C — GroupNorm over an (N, 1, HW, C) channel-last tensor.

    from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C
"""

from . import config
from .groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

__all__ = ["groupnorm_sc_N_1_HW_C", "config"]
