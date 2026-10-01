// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU half of the index score: folds 8-head custom_mm results into one 1x32 score row.
//
// custom_mm with an 8x32 query tile leaves an 8x32 result (row = head, column = token) in rows 0-7
// of faces 0 and 1 of its Dst tile; rows 8-15 of each face hold split-accumulation partials. An
// SFPU vector is 4 rows x 8 columns (lane = 8 * row + column), so the result is eight vectors: SFPU
// offsets {0, 1, 8, 9} for rows 0-3 and the same + 2 for rows 4-7.
//
// Per head group, weighted_relu_accumulate adds ReLU(s) * w_head into four 4x8 partial-sum vectors
// kept at offsets {0, 1, 8, 9} of the accumulator tile. reduce_partials_to_row0 then sums the four
// rows of each partial, leaving the finished scores in row 0 of faces 0 and 1.
//
// SFPTRANSP swaps the register index with the row index, within LREG0-3 and within LREG4-7 at the
// same time: new[i][8j + col] = old[j][8i + col]. sfpi::subvec_transp only models one of the two
// groups, so every transpose here is issued raw with no sfpi values live across it.

#pragma once

#include <cstdint>

namespace ckernel::sfpu {
constexpr std::uint32_t kLightningAccTile = 0;
constexpr std::uint32_t kLightningScoreTile = 1;
}  // namespace ckernel::sfpu

#ifdef TRISC_MATH
#include "ckernel_ops.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// w8 holds the fp32 bit patterns of the 8 head weights in the score tile; they are exact bf16.
template <bool first>
inline void lightning_weighted_relu_accumulate(const std::uint32_t* w8) {
    constexpr std::uint32_t acc = kLightningAccTile * 32;
    constexpr std::uint32_t score = kLightningScoreTile * 32;
    constexpr std::uint32_t slots[4] = {0, 1, 8, 9};
    // Rows 4-7 of the accumulator tile are otherwise unused; they hold the per-row weights.
    constexpr std::uint32_t w_lo_slot = acc + 2;
    constexpr std::uint32_t w_hi_slot = acc + 3;

    _llk_math_eltwise_sfpu_start_(0);

    // Broadcast w[r] into LREG r, then one transpose makes row r of LREG0 w[r] and row r of LREG4
    // w[4 + r].
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, w8[0] >> 16);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, w8[1] >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, w8[2] >> 16);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, w8[3] >> 16);
    TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, w8[4] >> 16);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, w8[5] >> 16);
    TT_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, w8[6] >> 16);
    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, w8[7] >> 16);
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, w_lo_slot * 2);
    TTI_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, w_hi_slot * 2);

    const sfpi::vFloat w_lo = sfpi::dst_reg[w_lo_slot];
    const sfpi::vFloat w_hi = sfpi::dst_reg[w_hi_slot];
#pragma GCC unroll 4
    for (std::uint32_t i = 0; i < 4; ++i) {
        const std::uint32_t s = slots[i];
        const sfpi::vFloat lo = sfpi::dst_reg[score + s];
        const sfpi::vFloat hi = sfpi::dst_reg[score + s + 2];
        sfpi::vFloat sum = 0.0f;
        if constexpr (!first) {
            sum = sfpi::dst_reg[acc + s];
        }
        v_if(lo > 0.0f) { sum += lo * w_lo; }
        v_endif;
        v_if(hi > 0.0f) { sum += hi * w_hi; }
        v_endif;
        sfpi::dst_reg[acc + s] = sum;
    }

    // custom_mm accumulates into Dst, so clear every row it writes (results and split partials)
    // before the next group's matmul.
#pragma GCC unroll 16
    for (std::uint32_t i = 0; i < 16; ++i) {
        sfpi::dst_reg[score + i] = 0.0f;
    }
    _llk_math_eltwise_sfpu_done_();
    // The next custom_mm writes the score tile this pass reads and clears.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

inline void lightning_reduce_partials_to_row0() {
    constexpr std::uint32_t base = kLightningAccTile * 64;

    _llk_math_eltwise_sfpu_start_(0);
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, base + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, base + 2);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, base + 16);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, base + 18);

    // Row j of every LREG now comes from partial j, so the lanewise sum holds partial j's column
    // sums in row j.
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    // Transposing back puts partial j's column sums in row 0 of LREG j.
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, base + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, base + 2);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, base + 16);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, base + 18);
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
