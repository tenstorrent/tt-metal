// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"

/**
 * @brief Copy one 32x32 tile from SrcA into DEST with eight MOVA2D, without a MOP.
 *
 * Uses ADDR_MOD_4 only ({srca=8, dest=8}), preserving matmul's ADDR_MOD 0,1,2 and reduce's ADDR_MOD 2,3.
 * ADDR_MOD_4 is a "don't care" -- matmul restores it on next reinit. Assumes a full 32x32 tile (4 faces,
 * 8 rows each).
 *
 * @note Sets the Src zero-substitution flag to the copy policy for the cached SrcA format (keep, so bf16
 *       -0.0 and denormals land in DEST unchanged; flush for fp8 sources), as @ref copy_tile_init does, and
 *       LEAVES it there. Every FPU compute init / reinit that may follow this copy re-applies the
 *       operand-driven default: @ref _llk_math_matmul_init_no_mop_ / @ref _llk_math_matmul_reinit_no_mop_,
 *       @ref _llk_math_reduce_block_max_row_init_ / @ref _llk_math_reduce_block_max_row_reinit_minimal_ /
 *       @ref _llk_math_reduce_block_max_row_reinit_with_mop_ and their runtime twins, and the
 *       custom_mm / sdpa_custom_mm / eltwise-binary inits.
 */
inline void _llk_math_eltwise_unary_datacopy_custom_(const std::uint32_t dest_index = 0)
{
    // MOVA2D flushes any SrcA datum with a zero exponent byte (bf16 -0.0, denormals) while the Src
    // zero-substitution flag holds the operand default. Apply the copy policy for the SrcA format
    // hw_configure / reconfig cached (keep, except fp8 whose zero carries a residual); skip-if-set.
    math::_configure_copy_zero_flag_state_(math::src_zero_flag_srca_fmt);

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
