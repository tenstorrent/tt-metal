# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY (NC.5): the pre-port Python prepared route of ttnn.bringup.pair_pool (see pair_pool.py), kept so
models/demos/deepseek_v41_flash_d_p/tests/native/test_native_lowbit.py can compare the native C++ op with it. Not
registered, not called by any model."""

from .pair_pool import pair_pool as pair_pool_generic

__all__ = ["pair_pool_generic"]
