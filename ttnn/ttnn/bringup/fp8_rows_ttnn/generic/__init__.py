# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY (NC.5): the pre-port Python prepared routes of ttnn.bringup.pack_fp8_rows / unpack_fp8_rows (see
fp8_rows.py), kept so models/demos/deepseek_v41_flash_d_p/tests/native/test_native_lowbit.py can compare the native C++
ops with them. Not registered, not called by any model."""

from .fp8_rows import pack_fp8_rows as pack_fp8_rows_generic
from .fp8_rows import unpack_fp8_rows as unpack_fp8_rows_generic

__all__ = ["pack_fp8_rows_generic", "unpack_fp8_rows_generic"]
