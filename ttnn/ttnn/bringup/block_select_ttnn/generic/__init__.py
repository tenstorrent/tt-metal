# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY (NC.6): the pre-port Python prepared routes of ttnn.bringup.block_gather / block_keys / block_max_bf16
(see block_select.py), kept so models/demos/deepseek_v41_flash_d_p/tests/native/test_native_block_select.py can compare
the native C++ ops with them. Not registered, not called by any model."""

from .block_select import block_gather as block_gather_generic
from .block_select import block_keys as block_keys_generic
from .block_select import block_max_bf16 as block_max_bf16_generic

__all__ = ["block_gather_generic", "block_keys_generic", "block_max_bf16_generic"]
