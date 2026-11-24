// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

namespace ckernel
{

inline void _llk_math_eltwise_mul_scalar_block_init_()
{
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Program ADDR_MOD_7 for the scalar-block walk.
 *
 * @note Call before a run of @ref _llk_math_eltwise_mul_scalar_block_ with AddrModScope::CallerScoped, and call
 *       @ref _llk_math_eltwise_mul_scalar_block_addrmod_restore_ after it.
 */
inline void _llk_math_eltwise_mul_scalar_block_addrmod_set_()
{
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_7);
}

/**
 * @brief Restore the ADDR_MOD_7 zero-advance invariant.
 *
 * @note Call after the last CallerScoped @ref _llk_math_eltwise_mul_scalar_block_ of a run, before any other math op.
 */
inline void _llk_math_eltwise_mul_scalar_block_addrmod_restore_()
{
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
}

/**
 * @brief Multiply a block of tiles by one reused scalar SrcB: Output[i] = SrcA[i] * SrcB.
 *
 * @tparam addrmod_scope PerCall: programs ADDR_MOD_7 for the walk and restores its zero-advance invariant on
 *         return. CallerScoped: leaves ADDR_MOD_7 alone; the caller brackets a run of calls with
 *         @ref _llk_math_eltwise_mul_scalar_block_addrmod_set_ and
 *         @ref _llk_math_eltwise_mul_scalar_block_addrmod_restore_, with no other math op in between.
 */
template <AddrModScope addrmod_scope = AddrModScope::PerCall>
inline void _llk_math_eltwise_mul_scalar_block_(const std::uint32_t dst_index, const std::uint32_t block_size)
{
    constexpr std::uint32_t kOpsPerTile = 8;

    if constexpr (addrmod_scope == AddrModScope::PerCall)
    {
        _llk_math_eltwise_mul_scalar_block_addrmod_set_();
    }

    for (std::uint32_t i = 0; i < block_size; ++i)
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + i);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
        for (std::uint32_t op = 0; op < kOpsPerTile; ++op)
        {
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, p_elwise::SRCB_BCAST_ALL, ADDR_MOD_7, 0);
        }
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
    }

    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
    math::clear_dst_reg_addr();

    if constexpr (addrmod_scope == AddrModScope::PerCall)
    {
        _llk_math_eltwise_mul_scalar_block_addrmod_restore_();
    }
}

} // namespace ckernel
