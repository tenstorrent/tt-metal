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
#include "llk_sfpu_types.h"
#include "lltt.h"

using namespace ckernel;

// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_binary_sfpu_configure_addrmod()
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

    if constexpr (
        sfpu_op == SfpuType::mul_int32 || sfpu_op == SfpuType::mul_uint16 || sfpu_op == SfpuType::max || sfpu_op == SfpuType::min ||
        sfpu_op == SfpuType::max_int32 || sfpu_op == SfpuType::min_int32 || sfpu_op == SfpuType::max_uint32 || sfpu_op == SfpuType::min_uint32 ||
        sfpu_op == SfpuType::lt_int || sfpu_op == SfpuType::gt_int || sfpu_op == SfpuType::le_int || sfpu_op == SfpuType::ge_int || sfpu_op == SfpuType::lt ||
        sfpu_op == SfpuType::gt || sfpu_op == SfpuType::le || sfpu_op == SfpuType::ge || sfpu_op == SfpuType::eq || sfpu_op == SfpuType::ne)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_6);
    }
}

inline void eltwise_binary_sfpu_configure_mop();

inline void _llk_math_eltwise_binary_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_binary_sfpu_done_()
{
    math::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_binary_sfpu_inc_dst_face_addr_()
{
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_binary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    eltwise_binary_sfpu_configure_addrmod<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_eltwise_binary_sfpu_uninit_()
{
    // No state to restore - all states are transient or default
}

// Shared per-tile replay loop for binary SFPU operations whose one-iteration
// body has been recorded into replay slot 0.
template <std::uint32_t REPLAY_LEN>
ALWI void _llk_replay_binary_sfpu_eltwise_(std::uint32_t idst0)
{
    _llk_math_eltwise_binary_sfpu_start_(idst0);

#pragma GCC unroll 0
    for (int face = 0; face < 4; face++)
    {
#pragma GCC unroll 0
        for (int d = 0; d < 8; d++)
        {
            lltt::replay(0, REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    _llk_math_eltwise_binary_sfpu_done_();
}
