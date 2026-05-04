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
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_sfpu_types.h"
#include "sfpu/ckernel_sfpu_welfords.h"

// local function declarations
inline void welfords_sfpu_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}

inline void welfords_sfpu_configure_mop();

template <DstSync Dst>
inline void _llk_math_welfords_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_welfords_sfpu_done_()
{
    math::clear_dst_reg_addr();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

inline void _llk_math_welfords_sfpu_inc_dst_face_addr_()
{
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

inline void _llk_math_welfords_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    welfords_sfpu_configure_addrmod();
    math::reset_counters(p_setrwc::SET_ABD_F);
    _program_welfords_replay_buffer_();
}

/**
 * Re-establish MATH address mods and MOP state for the SFPU Welford path after an
 * arbitrary MATH/FPU op (e.g. eltwise binary scalar multiply). Does not touch the
 * SFPU replay buffer or clear running mean/M2 in LREG4/5.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_math_welfords_sfpu_reinit_(const std::uint32_t num_faces, const std::uint32_t dst_format)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    eltwise_unary_configure_addrmod<DataCopyType::A2D>(dst_format);
    eltwise_unary_configure_mop<DataCopyType::A2D, is_fp32_dest_acc_en>(p_mova2d::MOV_8_ROWS, 16, num_faces, dst_format);
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_welfords_sfpu_uninit_()
{
    // No state to restore - all states are transient or default
}
