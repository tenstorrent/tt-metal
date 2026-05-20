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

// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_addrmod()
{
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    if constexpr (sfpu_op == SfpuType::topk_local_sort)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 32},
        }
            .set(ADDR_MOD_6);
    }

    if constexpr (
        sfpu_op == SfpuType::reciprocal || sfpu_op == SfpuType::typecast || sfpu_op == SfpuType::unary_max || sfpu_op == SfpuType::unary_min ||
        sfpu_op == SfpuType::unary_max_int32 || sfpu_op == SfpuType::unary_min_int32 || sfpu_op == SfpuType::unary_max_uint32 ||
        sfpu_op == SfpuType::unary_min_uint32 || sfpu_op == SfpuType::signbit || sfpu_op == SfpuType::equal_zero || sfpu_op == SfpuType::not_equal_zero ||
        sfpu_op == SfpuType::less_than_zero || sfpu_op == SfpuType::greater_than_equal_zero || sfpu_op == SfpuType::less_than_equal_zero ||
        sfpu_op == SfpuType::greater_than_zero)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_6);
    }
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}
