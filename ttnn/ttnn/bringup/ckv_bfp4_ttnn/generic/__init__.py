# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY (NC.5): the pre-port Python prepared route of ttnn.bringup.ckv_bfp4_rows (see ckv_bfp4.py), kept so
models/demos/deepseek_v41_flash_d_p/tests/native/test_native_lowbit.py can compare the native C++ op with it. Not
registered, not called by any model."""

from .ckv_bfp4 import ckv_bfp4_rows as ckv_bfp4_rows_generic

__all__ = ["ckv_bfp4_rows_generic"]
