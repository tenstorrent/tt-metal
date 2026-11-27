# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY (NC.1): the pre-port Python prepared route of ttnn.bringup.grouped_matmul (see grouped_matmul.py)."""

from .grouped_matmul import grouped_matmul as grouped_matmul_generic
from .grouped_matmul import supported

__all__ = ["grouped_matmul_generic", "supported"]
