// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"

// Uses ADDR_MOD_4 only ({srca=8, dest=8}), preserving matmul's ADDR_MOD 0,1,2
// and reduce's ADDR_MOD 2,3. ADDR_MOD_4 is a "don't care" — matmul restores it
// on next reinit. Assumes full 32x32 tile (4 faces, 8 rows each).
inline void _llk_math_eltwise_unary_datacopy_custom_(const std::uint32_t dest_index = 0)
{
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_4);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dest_index);

#pragma GCC unroll 8
    for (int i = 0; i < 8; i++)
    {
        TTI_MOVA2D(0, 0, ADDR_MOD_4, p_mova2d::MOV_8_ROWS, 0);
    }
    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
}
