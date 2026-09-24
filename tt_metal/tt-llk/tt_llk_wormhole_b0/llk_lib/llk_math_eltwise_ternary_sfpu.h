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
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu_types.h"

/**
 * @brief Legacy ADDR_MOD_6 table for the SfpuType-selected ternary SFPU init.
 *
 * Kept only for the deprecated @ref _llk_math_eltwise_ternary_sfpu_init_ path. New code programs
 * ADDR_MOD_6 in the op's own init instead.
 *
 * @tparam sfpu_op: SFPU operation whose ADDR_MOD_6 requirement is looked up
 */
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_ternary_sfpu_legacy_addrmod_()
{
    if constexpr (sfpu_op == SfpuType::where)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_6);
    }
}

/**
 * @brief SfpuType-selected ternary SFPU init (deprecated path).
 *
 * Runs the op-agnostic SFPU init and programs ADDR_MOD_6 from the legacy op table. New code calls
 * @ref _llk_math_eltwise_sfpu_init_ followed by the op's own init.
 *
 * @tparam sfpu_op: SFPU operation whose legacy ADDR_MOD_6 requirement is applied
 */
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_ternary_sfpu_init_()
{
    _llk_math_eltwise_sfpu_configure_common_();
    _llk_math_eltwise_ternary_sfpu_legacy_addrmod_<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}
