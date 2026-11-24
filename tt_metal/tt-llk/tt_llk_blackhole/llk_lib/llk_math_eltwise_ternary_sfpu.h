// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_eltwise_ternary_sfpu.h, ported except SFPU. Same Tensix instruction sequence as production.

#pragma once
#include <cstdint>
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu_types.h"

// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_ternary_sfpu_configure_addrmod()
{
    hal_port::set_sfpu_addr_mod<hal::cfg::Sec::S7 /*ADDR_MOD_7*/, 0>();

    if constexpr (sfpu_op == SfpuType::where)
    {
        hal_port::set_sfpu_addr_mod<hal::cfg::Sec::S6 /*ADDR_MOD_6*/, 2>();
    }
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_ternary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    eltwise_ternary_sfpu_configure_addrmod<sfpu_op>();
    hal_port::reset_counters_abd_f();
}
