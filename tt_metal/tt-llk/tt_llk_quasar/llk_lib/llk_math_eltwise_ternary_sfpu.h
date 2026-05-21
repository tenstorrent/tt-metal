// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"

// Init for ternary SFPU ops. Mirrors _llk_math_eltwise_unary_sfpu_init_ /
// _llk_math_eltwise_binary_sfpu_params_ — wraps the shared SFPU init that
// programs ADDR_MOD_7 (incr=0). Per-op state (e.g. ADDR_MOD_6 with dest.incr=2
// for `where`) is set up by the op's own `_init_<op>_` after this call,
// matching the Blackhole convention.
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_ternary_sfpu_init_()
{
    _llk_math_sfpu_init_();
}
