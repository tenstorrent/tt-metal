// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_math_common.h"

using namespace ckernel;

inline void _llk_math_eltwise_unary_datacopy_softmax_k_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // Copy row 0 from SrcB, then broadcast its first scalar into DEST row 8.
    TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_1_ROW, 0);
    TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_1_ROW_D0_BRCST, 0);
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0);

    math::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_unary_datacopy_softmax_k_init_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}
