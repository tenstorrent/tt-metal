// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU passes of the single-user hyperconnection program for a block of ROWS rows (ROWS in {2, 4, 8}): the streams,
// the sublayer output and collapsed are (ROWS x 32)-tiled blocks, whose bytes repeat a 16-column face row for each of
// the ROWS rows in turn. The collapse cores read their shard as "virtual" Hv x 32 tiles (Hv a multiple of ROWS), so a
// Dst vector's sub-row lr (lanes 8 lr .. 8 lr + 7) of vector i is block row
//   j = (4 * ((i >> 1) & 3) + lr) % ROWS,
// i.e. lr % ROWS for ROWS <= 4, and 4 g + lr with g = (i >> 1) & 1 for ROWS = 8 (G = 2 lane groups).
//
// Per-row weights are vectors whose sub-row lr holds row j's weight, prepared by the reader in a 32x32 bf16 tile (see
// reader_fused_single_user.cpp) that the compute copies into a Dst slot: weight vector v is dst_reg[32 * slot + v].
//
// See hc_single_user_sfpu.h for the SFPU vector layout.

#pragma once

#include <cstdint>

#include "hc_single_user_sfpu.h"

#ifdef TRISC_MATH

namespace ckernel::sfpu {

// Lane groups of a block of ROWS rows: one for ROWS <= 4, two for ROWS = 8.
template <std::uint32_t ROWS>
constexpr std::uint32_t hc_rows_groups() {
    return ROWS > 4 ? 2 : 1;
}

// Whether data vector i (of a virtual tile's faces 0 and 1) belongs to lane group g.
template <std::uint32_t G>
constexpr bool hc_rows_in_group(std::uint32_t i, std::uint32_t g) {
    return G == 1 || ((i >> 1) & 1u) == g;
}

// Dst slot 0 <- sum_k W(k) * Dst slot k over the first 16 vectors, W(k) the per-row weight vector k * G + g.
template <std::uint32_t H, std::uint32_t ROWS, std::uint32_t w_slot>
inline void hc_collapse_streams_rows() {
    static_assert(H >= 1 && H <= 4, "the collapse holds at most four streams in Dst");
    constexpr std::uint32_t G = hc_rows_groups<ROWS>();
    _llk_math_eltwise_sfpu_start_(0);
    constexpr std::uint32_t wb = 32 * w_slot;
#pragma GCC unroll 2
    for (std::uint32_t g = 0; g < G; ++g) {
        const sfpi::vFloat w0 = sfpi::dst_reg[wb + 0 * G + g];
        const sfpi::vFloat w1 = sfpi::dst_reg[wb + (H > 1 ? 1 : 0) * G + g];
        const sfpi::vFloat w2 = sfpi::dst_reg[wb + (H > 2 ? 2 : 0) * G + g];
        const sfpi::vFloat w3 = sfpi::dst_reg[wb + (H > 3 ? 3 : 0) * G + g];
#pragma GCC unroll 16
        for (std::uint32_t i = 0; i < 16; ++i) {
            if (!hc_rows_in_group<G>(i, g)) {
                continue;
            }
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
    }
    _llk_math_eltwise_sfpu_done_();
}

// hc_mix_collapse_streams for a block: Dst slots 0..H-1 the streams, slot H the sublayer output; per-row weights in
// w_slot: vector (r * (H + 1) + k) * G + g is comb[k][r] (k < H) / post[r] (k = H), vector (H * (H + 1) + r) * G + g
// is pre[r]. Leaves the mixed streams in slots 0..H-1 and the collapse in slot H.
template <std::uint32_t H, std::uint32_t ROWS, std::uint32_t w_slot>
inline void hc_mix_collapse_streams_rows() {
    static_assert(H >= 1 && H <= 4, "the mix holds at most four streams in Dst");
    constexpr std::uint32_t G = hc_rows_groups<ROWS>();
    _llk_math_eltwise_sfpu_start_(0);
    constexpr std::uint32_t wb = 32 * w_slot;
    auto wv = [&](std::uint32_t q, std::uint32_t g) -> std::uint32_t { return wb + q * G + g; };
#pragma GCC unroll 4
    for (std::uint32_t r = 0; r < H; ++r) {
#pragma GCC unroll 2
        for (std::uint32_t g = 0; g < G; ++g) {
            {
                const sfpi::vFloat wp = sfpi::dst_reg[wv(r * (H + 1) + H, g)];
#pragma GCC unroll 16
                for (std::uint32_t i = 0; i < 16; ++i) {
                    if (!hc_rows_in_group<G>(i, g)) {
                        continue;
                    }
                    sfpi::dst_reg[32 * r + 16 + i] = sfpi::dst_reg[32 * H + i] * wp;
                }
            }
            const sfpi::vFloat w0 = sfpi::dst_reg[wv(r * (H + 1) + 0, g)];
            const sfpi::vFloat w1 = sfpi::dst_reg[wv(r * (H + 1) + (H > 1 ? 1 : 0), g)];
            const sfpi::vFloat w2 = sfpi::dst_reg[wv(r * (H + 1) + (H > 2 ? 2 : 0), g)];
            const sfpi::vFloat w3 = sfpi::dst_reg[wv(r * (H + 1) + (H > 3 ? 3 : 0), g)];
#pragma GCC unroll 16
            for (std::uint32_t i = 0; i < 16; ++i) {
                if (!hc_rows_in_group<G>(i, g)) {
                    continue;
                }
                sfpi::vFloat acc = sfpi::dst_reg[32 * r + 16 + i];
                acc += sfpi::dst_reg[i] * w0;
                if constexpr (H > 1) {
                    acc += sfpi::dst_reg[32 + i] * w1;
                }
                if constexpr (H > 2) {
                    acc += sfpi::dst_reg[64 + i] * w2;
                }
                if constexpr (H > 3) {
                    acc += sfpi::dst_reg[96 + i] * w3;
                }
                sfpi::dst_reg[32 * r + 16 + i] = acc;
            }
        }
    }
#pragma GCC unroll 2
    for (std::uint32_t g = 0; g < G; ++g) {
        const sfpi::vFloat p0 = sfpi::dst_reg[wv(H * (H + 1) + 0, g)];
        const sfpi::vFloat p1 = sfpi::dst_reg[wv(H * (H + 1) + (H > 1 ? 1 : 0), g)];
        const sfpi::vFloat p2 = sfpi::dst_reg[wv(H * (H + 1) + (H > 2 ? 2 : 0), g)];
        const sfpi::vFloat p3 = sfpi::dst_reg[wv(H * (H + 1) + (H > 3 ? 3 : 0), g)];
#pragma GCC unroll 16
        for (std::uint32_t i = 0; i < 16; ++i) {
            if (!hc_rows_in_group<G>(i, g)) {
                continue;
            }
            sfpi::vFloat m = sfpi::dst_reg[16 + i];
            sfpi::dst_reg[i] = m;
            sfpi::vFloat acc = m * p0;
            if constexpr (H > 1) {
                m = sfpi::dst_reg[48 + i];
                sfpi::dst_reg[32 + i] = m;
                acc += m * p1;
            }
            if constexpr (H > 2) {
                m = sfpi::dst_reg[80 + i];
                sfpi::dst_reg[64 + i] = m;
                acc += m * p2;
            }
            if constexpr (H > 3) {
                m = sfpi::dst_reg[112 + i];
                sfpi::dst_reg[96 + i] = m;
                acc += m * p3;
            }
            sfpi::dst_reg[32 * H + i] = acc;
        }
    }
    _llk_math_eltwise_sfpu_done_();
}

// LREG0 <- each 8-lane sub-row's sum in all its lanes; then, with FOLD (ROWS < 4 on virtual tiles, whose sub-rows
// lr and lr + ROWS are the same block row), sub-row lr <- the sum over the sub-rows congruent to lr mod ROWS.
// Clobbers LREG1..7. Blackhole needs a NOP after each SUBVEC_SHFLROR1.
template <std::uint32_t ROWS, bool FOLD>
inline void hc_rows_lane_sum_lreg0_() {
#pragma GCC unroll 3
    for (std::uint32_t shift = 4; shift >= 1; shift /= 2) {
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        for (std::uint32_t s = 0; s < shift; ++s) {
            TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPNOP;
        }
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    }
    if constexpr (FOLD && ROWS < 4) {
        // Sub-row lr of LREG0 into sub-row 0 of LREG lr, fold, and back.
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPTRANSP(0, 0, 0, 0);
        if constexpr (ROWS == 2) {
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG1, 0);
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        } else {
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG0, 0);
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
        }
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
}

// Stores LREG0 to the four vectors of Dst row group g of a (rows x 32) tile in slot `slot`: faces 0 and 1, both column
// parities, so the tile's rows 4g .. 4g + 3 hold LREG0's sub-rows in every column.
inline void hc_rows_store_group_lreg0_(std::uint32_t slot, std::uint32_t g) {
    const std::uint32_t a = 64 * slot + 4 * g;
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, a);
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, a + 2);
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, a + 16);
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, a + 18);
}

// The fn dot's per-row sum(A^2): Dst slots 0..NT-1 hold (ROWS x 32) tiles of A (rows 0..ROWS-1 of faces 0 and 1, the
// rest zeroed by the unpacker). Writes slot 0 as a (ROWS x 32) tile whose row j holds row j's sum in every column.
template <std::uint32_t ROWS, std::uint32_t NT>
inline void hc_dot_rows_sumsq() {
    constexpr std::uint32_t G = hc_rows_groups<ROWS>();
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 2
    for (std::uint32_t g = 0; g < G; ++g) {
        sfpi::vFloat acc = 0.0f;
#pragma GCC unroll 8
        for (std::uint32_t t = 0; t < NT; ++t) {
            const sfpi::vFloat a = sfpi::dst_reg[32 * t + 2 * g];
            const sfpi::vFloat b = sfpi::dst_reg[32 * t + 2 * g + 1];
            const sfpi::vFloat c = sfpi::dst_reg[32 * t + 8 + 2 * g];
            const sfpi::vFloat d = sfpi::dst_reg[32 * t + 9 + 2 * g];
            acc += a * a + b * b + c * c + d * d;
        }
        // Parked in a vector of slot NT - 1's face 2, which no other group reads, for the raw section to pick up.
        sfpi::dst_reg[32 * (NT - 1) + 16 + g] = acc;
    }
#pragma GCC unroll 2
    for (std::uint32_t g = 0; g < G; ++g) {
        TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 64 * (NT - 1) + 32 + 2 * g);
        hc_rows_lane_sum_lreg0_<ROWS, false>();
        hc_rows_store_group_lreg0_(0, g);
    }
    _llk_math_eltwise_sfpu_done_();
}

// The collapse cores' per-row sum(x^2) over NT virtual (HV x 32) tiles of collapsed in Dst slots 0..NT-1, plus (CARRY)
// the earlier tiles' sums in slot carry_slot (a stat tile, see below). Writes slot out_slot as a (max(ROWS, 4) x 32)
// stat tile: row lr (and row 4 g + lr) holds the sum of the block row that sub-row is, in every column.
template <
    std::uint32_t ROWS,
    std::uint32_t HV,
    std::uint32_t nt,
    std::uint32_t out_slot,
    bool CARRY,
    std::uint32_t carry_slot>
inline void hc_rows_virtual_sumsq() {
    constexpr std::uint32_t G = hc_rows_groups<ROWS>();
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 2
    for (std::uint32_t g = 0; g < G; ++g) {
        sfpi::vFloat acc = 0.0f;
#pragma GCC unroll 8
        for (std::uint32_t t = 0; t < nt; ++t) {
#pragma GCC unroll 16
            for (std::uint32_t i = 0; i < 16; ++i) {
                // Only a face's first HV rows hold the tile; the unpacker leaves the rest of the face undefined.
                if (!hc_rows_in_group<G>(i, g) || ((i & 7u) >> 1) >= HV / 4) {
                    continue;
                }
                const sfpi::vFloat x = sfpi::dst_reg[32 * t + i];
                acc += x * x;
            }
        }
        sfpi::dst_reg[32 * out_slot + 16 + g] = acc;
    }
#pragma GCC unroll 2
    for (std::uint32_t g = 0; g < G; ++g) {
        TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 64 * out_slot + 32 + 2 * g);
        hc_rows_lane_sum_lreg0_<ROWS, true>();
        if constexpr (CARRY) {
            TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 64 * carry_slot + 4 * g);
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        }
        hc_rows_store_group_lreg0_(out_slot, g);
    }
    _llk_math_eltwise_sfpu_done_();
}

// The fused norm's last step on NT virtual tiles: slot t (t < NT) <- slot t * scale(row) * slot (gamma_slot + t), the
// scale read from the stat-shaped tile in scale_slot (row group g's vector 2 g).
template <std::uint32_t ROWS, std::uint32_t nt, std::uint32_t gamma_slot, std::uint32_t scale_slot>
inline void hc_rows_scale_gamma() {
    constexpr std::uint32_t G = hc_rows_groups<ROWS>();
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 2
    for (std::uint32_t g = 0; g < G; ++g) {
        const sfpi::vFloat s = sfpi::dst_reg[32 * scale_slot + 2 * g];
#pragma GCC unroll 8
        for (std::uint32_t t = 0; t < nt; ++t) {
#pragma GCC unroll 16
            for (std::uint32_t i = 0; i < 16; ++i) {
                if (!hc_rows_in_group<G>(i, g)) {
                    continue;
                }
                sfpi::dst_reg[32 * t + i] = sfpi::dst_reg[32 * t + i] * s * sfpi::dst_reg[32 * (gamma_slot + t) + i];
            }
        }
    }
    _llk_math_eltwise_sfpu_done_();
}

// post / pre for ROWS rows at once, elementwise: Dst tile 0 holds post_w (face 0), post_b (face 1), pre_w (face 2) and
// pre_b (face 3), each at (row k, column j) for stream k and block row j (zero elsewhere). Writes Dst tile 1:
//   face 0 = 2 * sigmoid(post_w * post_scale + post_b),  face 2 = sigmoid(pre_w * pre_scale + pre_b) + eps.
template <std::uint32_t H, std::uint32_t POST_SCALE_BITS, std::uint32_t PRE_SCALE_BITS, std::uint32_t EPS_BITS>
inline void hc_post_pre_rows() {
    _llk_math_eltwise_sfpu_start_(0);
    // _sfpu_sigmoid_'s reciprocal reads 2.0 from here.
    sfpi::vConstFloatPrgm0 = 2.0f;
    // Rows k < H of a face are its first 2 * ceil(H / 4) vectors.
    constexpr std::uint32_t num_vectors = 2 * ((H + 3) / 4);
    const sfpi::vFloat post_scale = Converter::as_float(POST_SCALE_BITS);
    const sfpi::vFloat pre_scale = Converter::as_float(PRE_SCALE_BITS);
    const sfpi::vFloat eps = Converter::as_float(EPS_BITS);
#pragma GCC unroll 8
    for (std::uint32_t i = 0; i < num_vectors; ++i) {
        const sfpi::vFloat x = sfpi::dst_reg[i] * post_scale + sfpi::dst_reg[8 + i];
        sfpi::dst_reg[32 + i] = 2.0f * _sfpu_sigmoid_<true>(x);
        const sfpi::vFloat y = sfpi::dst_reg[16 + i] * pre_scale + sfpi::dst_reg[24 + i];
        sfpi::dst_reg[48 + i] = _sfpu_sigmoid_<true>(y) + eps;
    }
    _llk_math_eltwise_sfpu_done_();
}

// hc_comb_sinkhorn for ROWS rows at once: block row j's comb_w (face 0) and comb_b (face 1) sit at face row
// r + 4 * (c / 2), face column 2 j + c % 2 -- column c is vector c, with (row r, block row j) in lane 8 r + j -- and
// every step is lanewise or an SFPTRANSP, which keeps a lane's place in its sub-row. Leaves column c of every row's
// [H, H] result in vector c, in the same lanes.
template <std::uint32_t H, std::uint32_t ROWS, std::uint32_t ITERS, std::uint32_t EPS_BITS, std::uint32_t SCALE_BITS>
inline void hc_comb_sinkhorn_rows() {
    static_assert(H >= 1 && H <= 4 && ITERS >= 1 && ROWS >= 1 && ROWS <= 8);
    _llk_math_eltwise_sfpu_start_(0);

    const sfpi::vFloat scale = Converter::as_float(SCALE_BITS);
#pragma GCC unroll 4
    for (std::uint32_t c = 0; c < 4; ++c) {
        sfpi::dst_reg[c] = sfpi::dst_reg[c] * scale + sfpi::dst_reg[8 + c];
    }
    sfpi::vFloat row_max = sfpi::dst_reg[0];
#pragma GCC unroll 3
    for (std::uint32_t c = 1; c < H; ++c) {
        row_max = sfpi::min_max(row_max, sfpi::vFloat(sfpi::dst_reg[c])).second;
    }
    sfpi::dst_reg[4] = row_max;
    const sfpi::vFloat eps = Converter::as_float(EPS_BITS);
#pragma GCC unroll 4
    for (std::uint32_t c = 0; c < 4; ++c) {
        sfpi::vFloat e = 0.0f;
        sfpi::vFloat e_eps = 0.0f;
        if (c < H) {
            // vConstTileId is 2 * lane: lane 8 r + j holds block row j < ROWS of row r < H.
            v_if(
                (sfpi::vConstTileId & sfpi::vInt(0xE)) < static_cast<std::int32_t>(2 * ROWS) &&
                sfpi::vConstTileId < static_cast<std::int32_t>(16 * H)) {
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
    TTI_SFPTRANSP(0, 0, 0, 0);
#pragma GCC unroll 0
    for (std::uint32_t i = 0; i < 2 * ITERS - 1; ++i) {
        hc_sinkhorn_norm_rows_();
        TTI_SFPTRANSP(0, 0, 0, 0);
    }
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 4);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 6);
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace ckernel::sfpu

#endif  // TRISC_MATH
