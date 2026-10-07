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

using namespace ckernel;

// local function declarations
/**
 * @brief Program the op-specific SFPU address mod: ADDR_MOD_6 for the ops whose kernel steps dest by something other
 *        than the invariant zero of ADDR_MOD_7 (32 rows for topk_local_sort, 2 rows for the typecast / compare family).
 *
 * No-op for every other op. The kernel-invariant ADDR_MOD_7 is programmed by @ref _llk_math_eltwise_unary_sfpu_init_once_.
 *
 * @tparam sfpu_op: SFPU op being initialised.
 */
template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_op_addrmod()
{
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which uses ADDR_MOD_2 (and ADDR_MOD_3 for broadcasts), so use one
    //       that doesn't conflict!
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
        sfpu_op == SfpuType::typecast || sfpu_op == SfpuType::unary_max || sfpu_op == SfpuType::unary_min || sfpu_op == SfpuType::unary_max_int32 ||
        sfpu_op == SfpuType::unary_min_int32 || sfpu_op == SfpuType::unary_max_uint32 || sfpu_op == SfpuType::unary_min_uint32 ||
        sfpu_op == SfpuType::signbit || sfpu_op == SfpuType::not_equal_zero || sfpu_op == SfpuType::equal_zero || sfpu_op == SfpuType::less_than_zero ||
        sfpu_op == SfpuType::greater_than_equal_zero || sfpu_op == SfpuType::greater_than_zero || sfpu_op == SfpuType::less_than_equal_zero)
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
 * @brief Kernel-invariant half of the unary SFPU init: the SFPU config register (SFPCONFIG(0, 0xF, 1)) and the
 *        invariant ADDR_MOD_7 = {srca:0, srcb:0, dest:0}.
 *
 * Identical for every SFPU op, so it only needs to run once per kernel. Metal wires it into every "full init"
 * entry point (compute_kernel_hw_startup, init_sfpu, unary_op_init_common, binary_op_init_common) and into the
 * head of its bare per-op init, and the tt-llk standalone SFPU test harness wires it into its init prelude.
 *
 * @note Follow with @ref _llk_math_eltwise_unary_sfpu_init_residual_ for the op-specific part;
 *       @ref _llk_math_eltwise_unary_sfpu_init_ runs both.
 */
inline void _llk_math_eltwise_unary_sfpu_init_once_()
{
    sfpu::_init_sfpu_config_reg();

    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which uses ADDR_MOD_2 (and ADDR_MOD_3 for broadcasts), so use one
    //       that doesn't conflict!
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}

/**
 * @brief Op-specific half of the unary SFPU init: the op's ADDR_MOD_6 (where the op needs one) and the RWC counter reset.
 *
 * Everything @ref _llk_math_eltwise_unary_sfpu_init_ does beyond @ref _llk_math_eltwise_unary_sfpu_init_once_. A caller
 * that has already run the once-init (metal's bare per-op init entry point runs it for every op) uses this instead of
 * the full init, so SFPCONFIG and ADDR_MOD_7 are not programmed twice per `*_tile_init`.
 *
 * @tparam sfpu_op: SFPU op being initialised.
 * @note Call @ref _llk_math_eltwise_unary_sfpu_init_once_ (once per kernel is enough) before this function.
 */
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_residual_()
{
    eltwise_unary_sfpu_configure_op_addrmod<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Full unary SFPU init for one op: @ref _llk_math_eltwise_unary_sfpu_init_once_ followed by
 *        @ref _llk_math_eltwise_unary_sfpu_init_residual_.
 *
 * @tparam sfpu_op: SFPU op being initialised.
 */
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    _llk_math_eltwise_unary_sfpu_init_once_();
    _llk_math_eltwise_unary_sfpu_init_residual_<sfpu_op>();
}
