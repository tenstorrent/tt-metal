// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU passes of the single-user hyperconnection program. All of them run with an FP32 Dest, so
// every intermediate stays fp32 and only the final pack rounds to bf16.
//
// An SFPU vector is 4 rows x 8 columns of one face: sfpi::dst_reg[2*r4 + p] holds rows
// [4*r4, 4*r4 + 4) of face 0 at column parity p (lane = 8 * row + column / 2), faces follow at
// dst_reg offsets 8, 16 and 24, and Dst tile t starts at dst_reg[32 * t]. vConstTileId is 2 * lane.

#pragma once

#include <cstdint>

#ifdef TRISC_MATH
#include "ckernel_ops.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel::sfpu {

// Dst tile 0 <- sum_k w[k * W_STRIDE] * Dst tile k, over the first 16 vectors (faces 0 and 1) of
// each tile: every h x 32 tile with h <= 16 lies there. w points at the bf16 weights.
template <std::uint32_t H, std::uint32_t W_STRIDE>
inline void hc_collapse_streams(const volatile tt_l1_ptr std::uint16_t* w) {
    static_assert(H >= 1 && H <= 4, "the collapse holds at most four streams in Dst");
    _llk_math_eltwise_sfpu_start_(0);
    auto weight = [&](std::uint32_t k) {
        return Converter::as_float(static_cast<std::uint32_t>(w[k * W_STRIDE]) << 16);
    };
    const sfpi::vFloat w0 = weight(0);
    const sfpi::vFloat w1 = weight(H > 1 ? 1 : 0);
    const sfpi::vFloat w2 = weight(H > 2 ? 2 : 0);
    const sfpi::vFloat w3 = weight(H > 3 ? 3 : 0);
#pragma GCC unroll 16
    for (std::uint32_t i = 0; i < 16; ++i) {
        sfpi::vFloat acc = sfpi::dst_reg[i] * w0;
        if constexpr (H > 1) {
            acc += sfpi::dst_reg[32 + i] * w1;
        }
        if constexpr (H > 2) {
            acc += sfpi::dst_reg[64 + i] * w2;
        }
        if constexpr (H > 3) {
            acc += sfpi::dst_reg[96 + i] * w3;
        }
        sfpi::dst_reg[i] = acc;
    }
    _llk_math_eltwise_sfpu_done_();
}

// pre = sigmoid(pre_w * pre_scale + pre_b) + eps for an h x 32 Dst tile 0 whose row 0 interleaves
// pre_w[k] (column 2k) and pre_b[k] (column 2k + 1), k < 8: the even- and odd-column vectors line up
// w and b lane for lane, and pre[k] is left at column 2k.
template <std::uint32_t PRE_SCALE_BITS, std::uint32_t EPS_BITS>
inline void hc_pre_interleaved() {
    _llk_math_eltwise_sfpu_start_(0);
    // _sfpu_sigmoid_'s reciprocal reads 2.0 from here.
    sfpi::vConstFloatPrgm0 = 2.0f;
    sfpi::dst_reg[0] = _sfpu_sigmoid_<true>(sfpi::dst_reg[0] * Converter::as_float(PRE_SCALE_BITS) + sfpi::dst_reg[1]) +
                       Converter::as_float(EPS_BITS);
    _llk_math_eltwise_sfpu_done_();
}

// Dst tile 0 holds post_w / post_b in column 0 of faces 0 / 1 (rows 0..H-1) and pre_w / pre_b in
// row 0 of faces 2 / 3 (columns 0..H-1), everything else zero. Writes
//   Dst tile 1 = post = 2 * sigmoid(post_w * post_scale + post_b)  as an [H, 1] column (WITH_POST),
//   Dst tile 2 = pre  = sigmoid(pre_w * pre_scale + pre_b) + eps   as a [1, H] row (WITH_PRE),
// zero outside the logical rectangle.
template <
    std::uint32_t H,
    std::uint32_t POST_SCALE_BITS,
    std::uint32_t PRE_SCALE_BITS,
    std::uint32_t EPS_BITS,
    bool WITH_POST,
    bool WITH_PRE>
inline void hc_post_pre() {
    _llk_math_eltwise_sfpu_start_(0);
    // _sfpu_sigmoid_'s reciprocal reads 2.0 from here.
    sfpi::vConstFloatPrgm0 = 2.0f;
#pragma GCC unroll 32
    for (std::uint32_t i = 0; i < 32; ++i) {
        if constexpr (WITH_POST) {
            sfpi::dst_reg[32 + i] = 0.0f;
        }
        if constexpr (WITH_PRE) {
            sfpi::dst_reg[64 + i] = 0.0f;
        }
    }

    if constexpr (WITH_POST) {
        const sfpi::vFloat x = sfpi::dst_reg[0] * Converter::as_float(POST_SCALE_BITS) + sfpi::dst_reg[8];
        const sfpi::vFloat post = 2.0f * _sfpu_sigmoid_<true>(x);
        v_if(sfpi::vConstTileId < static_cast<std::int32_t>(16 * H)) {
            v_if((sfpi::vConstTileId & sfpi::vInt(0xE)) == 0) { sfpi::dst_reg[32] = post; }
            v_endif;
        }
        v_endif;
    }

    if constexpr (WITH_PRE) {
        const sfpi::vFloat eps = Converter::as_float(EPS_BITS);
        const sfpi::vFloat scale = Converter::as_float(PRE_SCALE_BITS);
        // Row 0 is lanes 0..7: even columns 2*lane, odd columns 2*lane + 1.
        const sfpi::vFloat even = _sfpu_sigmoid_<true>(sfpi::dst_reg[16] * scale + sfpi::dst_reg[24]) + eps;
        v_if(sfpi::vConstTileId < static_cast<std::int32_t>(H)) { sfpi::dst_reg[64] = even; }
        v_endif;
        const sfpi::vFloat odd = _sfpu_sigmoid_<true>(sfpi::dst_reg[17] * scale + sfpi::dst_reg[25]) + eps;
        v_if(sfpi::vConstTileId < static_cast<std::int32_t>(H) - 1) { sfpi::dst_reg[65] = odd; }
        v_endif;
    }
    _llk_math_eltwise_sfpu_done_();
}

// LREG0..3 hold the columns of a 4 x 4 matrix, row r in lane 8r (every other lane zero). Scales each
// row by 1 / (its sum + eps), eps in LREG13; LREG4/5 are scratch. On the SFPTRANSP'd registers the
// same code normalizes the columns.
inline void hc_sinkhorn_norm_rows_() {
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, 0);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG5, 0);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG13, p_sfpu::LREG4, 0);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    // SFPARECIP is a ~7-bit seed x; one Newton step 1/a = x + x * r with r = 1 - a * x.
    TTI_SFPARECIP(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG4, 1);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
}

// comb = sinkhorn(softmax(comb_w * comb_scale + comb_b) + eps) for H <= 4, all of it in fp32 LRegs.
// Dst tile 0 holds the H x H comb_w (face 0) and comb_b (face 1) with column c of row r at face row
// r + 4 * (c / 2), face column c % 2 -- so each column is one SFPU vector with row r in lane 8r --
// and zeros elsewhere. The [H, H] result is left at rows 0..H-1, columns 0..H-1 of face 0, the rest
// of the tile zeroed.
//
// Row sums are then lanewise adds of the four column vectors, and one SFPTRANSP (LReg index <-> lane
// group) makes column sums the same adds. Matches the reference step for step: softmax (max-shifted
// exp masked to the block, then a row-norm), + eps, column-norm, then ITERS - 1 row / column-norm pairs.
template <std::uint32_t H, std::uint32_t ITERS, std::uint32_t EPS_BITS, std::uint32_t SCALE_BITS>
inline void hc_comb_sinkhorn() {
    static_assert(H >= 1 && H <= 4 && ITERS >= 1);
    _llk_math_eltwise_sfpu_start_(0);

    const sfpi::vFloat scale = Converter::as_float(SCALE_BITS);
#pragma GCC unroll 4
    for (std::uint32_t c = 0; c < 4; ++c) {
        sfpi::dst_reg[c] = sfpi::dst_reg[c] * scale + sfpi::dst_reg[8 + c];
    }
    // Each row's max over its H columns, so the exp below cannot overflow. Kept in Dst: the exp needs
    // most of the LRegs. Nothing outside the H x H block is read, so the input tile's padding is
    // undefined.
    sfpi::vFloat row_max = sfpi::dst_reg[0];
#pragma GCC unroll 3
    for (std::uint32_t c = 1; c < H; ++c) {
        row_max = sfpi::min_max(row_max, sfpi::vFloat(sfpi::dst_reg[c])).second;
    }
    sfpi::dst_reg[4] = row_max;
    // Column c's valid entries: lane 8r (vConstTileId 16r) for r < H, and only for c < H. Face 2 takes
    // the + eps term, zero outside the block.
    const sfpi::vFloat eps = Converter::as_float(EPS_BITS);
#pragma GCC unroll 4
    for (std::uint32_t c = 0; c < 4; ++c) {
        sfpi::vFloat e = 0.0f;
        sfpi::vFloat e_eps = 0.0f;
        if (c < H) {
            v_if(
                (sfpi::vConstTileId & sfpi::vInt(0xE)) == 0 && sfpi::vConstTileId < static_cast<std::int32_t>(16 * H)) {
                e = _sfpu_exp_fp32_accurate_<false>(sfpi::vFloat(sfpi::dst_reg[c]) - sfpi::dst_reg[4]);
                e_eps = eps;
            }
            v_endif;
        }
        sfpi::dst_reg[c] = e;
        sfpi::dst_reg[16 + c] = e_eps;
    }
    sfpi::vConstFloatPrgm1 = eps;

    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 6);
    hc_sinkhorn_norm_rows_();
    TTI_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, 32);
    TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, 34);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
    TTI_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, 36);
    TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, 38);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG2, 0);
    TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG3, 0);
    // Column-norm, then ITERS - 1 (row, column) pairs: every normalization is a row-norm, alternately
    // of the matrix and of its transpose.
    TTI_SFPTRANSP(0, 0, 0, 0);
#pragma GCC unroll 0
    for (std::uint32_t i = 0; i < 2 * ITERS - 1; ++i) {
        hc_sinkhorn_norm_rows_();
        TTI_SFPTRANSP(0, 0, 0, 0);
    }

    // Back to [H, H] at face 0's top left: columns 2 and 3 move one lane right (lane 8r + 1 is face
    // column 2 / 3 at parity 0 / 1), onto lanes that are zero in columns 0 and 1.
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
#pragma GCC unroll 30
    for (std::uint32_t i = 2; i < 32; ++i) {
        sfpi::dst_reg[i] = 0.0f;
    }
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
