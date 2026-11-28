# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY (NC.2): the pre-port Python prepared route of ttnn.bringup.fabric_reduce_scatter (see
fabric_reduce_scatter.py), kept so models/demos/deepseek_v41_flash_d_p/tests/native/test_native_reduce_scatter.py can
compare the native C++ op with it. Not registered, not called by any model."""

from .fabric_reduce_scatter import fabric_reduce_scatter as fabric_reduce_scatter_generic
from .fabric_reduce_scatter import prepare

__all__ = ["fabric_reduce_scatter_generic", "prepare"]
