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
 * @param dest_index: Tile index in DEST that the copy is written to.
 * @param src_format: Data format of the copied operand as unpacked into SrcA (the CB's unpack destination format,
 *                    which copy_init derives from the CB). Keyed on the copied operand rather than on the cached
 *                    math-side SrcA format: the legacy custom_mm / compressed_custom_mm / sdpa_custom_mm compute
 *                    inits configure unpack as (in1, in0) but math as (in0, in1), so after one of them the math
 *                    cache can hold the other operand's format.
 * @note Sets the Src zero-substitution flag to the copy policy for src_format (keep, so bf16 -0.0 and denormals
 *       land in DEST unchanged; flush for fp8 sources), as copy_init does, and LEAVES it there. The inits / reinits
 *       that re-apply the operand-driven default after this copy are @ref _llk_math_matmul_init_no_mop_ /
 *       @ref _llk_math_matmul_reinit_no_mop_, @ref _llk_math_reduce_block_max_row_init_ /
 *       @ref _llk_math_reduce_block_max_row_reinit_minimal_ / @ref _llk_math_reduce_block_max_row_reinit_with_mop_
 *       and their runtime twins, the custom_mm / compressed_custom_mm / sdpa_custom_mm / sdpa_custom_mm_reuse_dest_srcb /
 *       face_compressed_mm / hadamard_h128 inits and the eltwise-binary init. The ones that do NOT are
 *       @ref _llk_math_sdpa_bcast_col_srcb_reuse_init_, @ref _llk_math_sdpa_bcast_col_srca_srcb_reuse_init_ and
 *       @ref _llk_math_custom_mm_reuse_dest_srcb_init_: an FPU op run through one of those after this copy keeps
 *       denormal Src operands.
 */
inline void _llk_math_eltwise_unary_datacopy_custom_(const std::uint32_t dest_index, const std::uint32_t src_format)
{
    // MOVA2D flushes any SrcA datum with a zero exponent byte (bf16 -0.0, denormals) while the Src
    // zero-substitution flag holds the operand default. Apply the copy policy for the copied operand's format
    // (keep, except fp8 whose zero carries a residual); skip-if-set.
    math::_configure_copy_zero_flag_state_(src_format);

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
