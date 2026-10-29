// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu_types.h"

using namespace ckernel;

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_binary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    math::reset_counters(p_setrwc::SET_ABD_F);
}
