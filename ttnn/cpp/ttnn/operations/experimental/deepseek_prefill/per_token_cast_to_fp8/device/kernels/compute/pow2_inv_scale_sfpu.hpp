// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp8 quantize helpers shared by per_token_cast_to_fp8 (FAST_1X32), the routed expert's Q2 epilogue and
// combine_fabric2d's quantizing untilize: from a row-max tile (column 0, one value per tile row), write the exact
// power-of-two 1/scale in place, and round the scaled values to the e4m3 grid.
#pragma once

#include <cstdint>

#include "api/compute/common.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// 1x32 pow2 scale from a row-max tile, in one pass on column 0: with
// scale = 2^e = 2^ceil(log2(max(amax, clamp_min) / 448)), write the exact 1/scale = 2^-e in place. The column-
// broadcast multiply takes it as is; the writer recovers the scale bits as (254 << 23) - (1/scale bits).
template <int ITERATIONS = 8>
inline void calculate_pow2_inv_scale(uint32_t clamp_min_bits, uint32_t inv_448_bits) {
    const sfpi::vFloat clamp_min = Converter::as_float(clamp_min_bits);
    const sfpi::vFloat inv_448 = Converter::as_float(inv_448_bits);
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v < clamp_min) { v = clamp_min; }
        v_endif;
        v = v * inv_448;
        sfpi::vInt e = sfpi::exexp(v, sfpi::ExponentMode::Biased);
        v_if(sfpi::setexp(v, 127) != 1.0f) { e = e + 1; }
        v_endif;
        sfpi::dst_reg[0] = sfpi::setexp(sfpi::vFloat(1.0f), 254 - e);
        sfpi::dst_reg++;
    }
}

// Fused |x| -> row max -> pow2 1/scale for one 32x32 tile, kept in registers: per 8-row group, load the 4 + 4 row
// slices of both faces with |.|, fold left / right faces and the 8 SFPU columns (the row max of |x| ends up in every
// column), then turn it into 1/scale with integer ops and store it over the group's left-face rows (column 0
// included). 1/scale = 2^-ceil(log2(max(amax, clamp) / 448)) has bits (254 << 23) - ((bits(v) + 0x7FFFFF) &
// 0x7F800000) for v = max(amax, clamp) * (1/448) > 0, bit-exact with calculate_pow2_inv_scale. The rest of the
// tile keeps x (the multiply re-reads x from L1). Replaces abs_tile + sfpu_reduce<MAX, REDUCE_ROW> +
// pow2_inv_scale_tile: 4 vectors of pow2 math per tile instead of 16, and no |x| pass over DEST.
inline void load_const_lreg(uint32_t lreg, uint32_t bits) {
    TT_SFPLOADI(lreg, SFPLOADI_MOD0_LOWER, bits & 0xFFFF);
    TT_SFPLOADI(lreg, SFPLOADI_MOD0_UPPER, bits >> 16);
}

inline void pow2_inv_scale_lreg(uint32_t r, uint32_t clamp_lreg) {
    TTI_SFPSWAP(0, r, clamp_lreg, 1);                      // r = max(r, clamp) (MAX direction)
    TTI_SFPMAD(r, p_sfpu::LREG2, p_sfpu::LCONST_0, r, 0);  // r *= 1/448
    TTI_SFPNOP;
    TTI_SFPIADD(0, p_sfpu::LREG3, r, sfpi::SFPIADD_MOD1_CC_NONE);  // + 0x7FFFFF: round the exponent up
    TTI_SFPAND(0, p_sfpu::LREG5, r, 0);                            // & 0x7F800000: biased ceil exponent
    TTI_SFPIADD(
        0, p_sfpu::LREG6, r, sfpi::SFPIADD_MOD1_CC_NONE | sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);  // (254<<23) - r
}

template <bool kPow2>
inline void amax_rows_tile(uint32_t clamp_min_bits, uint32_t inv_448_bits);

// Round fp32 to the e4m3 mantissa grid (3 explicit bits) with round-to-nearest-even, so the
// truncating fp8 packer then sees exact values. Same "add half-ulp - 1 + lsb, clear" bit trick as
// float32_to_bf16_rne, at bit 20 instead of 16. Inputs are already scaled to |x| <= 448, so no
// rounding reaches 480 (NaN in e4m3fn). e4m3 subnormals (|x| < 2^-6) keep packer truncation below
// their coarser grid; at these scales that is < 2^-14.8 of the block amax.
template <int ITERATIONS = 8>
inline void calculate_round_to_e4m3_rne() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(value);
        sfpi::vUInt lsb = (bits >> 20) & 1;
        bits = bits + 0x7ffffU + lsb;
        bits = bits & 0xfff00000U;
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(bits);
        sfpi::dst_reg++;
    }
}

inline void calculate_amax_pow2_inv_scale(uint32_t clamp_min_bits, uint32_t inv_448_bits) {
    amax_rows_tile<true>(clamp_min_bits, inv_448_bits);
}

// Row max of |x| only, stored like the 1/scale above (left-face rows, every SFPU column): input of the 1x128 fold.
inline void calculate_amax_rows(uint32_t, uint32_t) { amax_rows_tile<false>(0, 0); }

// 1x128 (four tiles share a scale per row): fold the row maxima the amax pass left in tiles 0..3 of the DEST batch
// (ROWS_PER_TILE apart), turn the max into the pow2 1/scale and store it in tile 0. Call with idst = first tile.
inline void calculate_fold4_pow2_inv_scale(uint32_t clamp_min_bits, uint32_t inv_448_bits) {
    constexpr uint32_t mode = static_cast<uint32_t>(InstrModLoadStore::FP32);
    set_sfpswap_direction<false>();
#pragma GCC unroll 2
    for (uint32_t face_pair = 0; face_pair < 2; ++face_pair) {
        const uint32_t base = face_pair * 2 * ROWS_PER_FACE;
#pragma GCC unroll 2
        for (uint32_t row_group = 0; row_group < 2; ++row_group) {
            const uint32_t r0 = base + row_group * 8;
            const uint32_t r1 = r0 + 4;
            TT_SFPLOAD(p_sfpu::LREG0, mode, ADDR_MOD_7, r0);
            TT_SFPLOAD(p_sfpu::LREG1, mode, ADDR_MOD_7, r0 + ROWS_PER_TILE);
            TT_SFPLOAD(p_sfpu::LREG2, mode, ADDR_MOD_7, r0 + 2 * ROWS_PER_TILE);
            TT_SFPLOAD(p_sfpu::LREG3, mode, ADDR_MOD_7, r0 + 3 * ROWS_PER_TILE);
            TT_SFPLOAD(p_sfpu::LREG4, mode, ADDR_MOD_7, r1);
            TT_SFPLOAD(p_sfpu::LREG5, mode, ADDR_MOD_7, r1 + ROWS_PER_TILE);
            TT_SFPLOAD(p_sfpu::LREG6, mode, ADDR_MOD_7, r1 + 2 * ROWS_PER_TILE);
            TT_SFPLOAD(p_sfpu::LREG7, mode, ADDR_MOD_7, r1 + 3 * ROWS_PER_TILE);
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, 1);
            load_const_lreg(p_sfpu::LREG1, clamp_min_bits);
            load_const_lreg(p_sfpu::LREG7, clamp_min_bits);
            load_const_lreg(p_sfpu::LREG2, inv_448_bits);
            load_const_lreg(p_sfpu::LREG3, 0x007FFFFFu);
            load_const_lreg(p_sfpu::LREG5, 0x7F800000u);
            load_const_lreg(p_sfpu::LREG6, 254u << 23);
            pow2_inv_scale_lreg(p_sfpu::LREG0, p_sfpu::LREG1);
            pow2_inv_scale_lreg(p_sfpu::LREG4, p_sfpu::LREG7);
            TT_SFPSTORE(p_sfpu::LREG0, mode, ADDR_MOD_7, r0);
            TT_SFPSTORE(p_sfpu::LREG4, mode, ADDR_MOD_7, r1);
        }
    }
}

template <bool kPow2>
inline void amax_rows_tile(uint32_t clamp_min_bits, uint32_t inv_448_bits) {
    constexpr uint32_t mode = static_cast<uint32_t>(InstrModLoadStore::FP32);
    set_sfpswap_direction<false>();  // SFPSWAP keeps the max in its first operand
#pragma GCC unroll 2
    for (uint32_t face_pair = 0; face_pair < 2; ++face_pair) {
        const uint32_t base = face_pair * 2 * ROWS_PER_FACE;
#pragma GCC unroll 2
        for (uint32_t row_group = 0; row_group < 2; ++row_group) {
            const uint32_t r0 = base + row_group * 8;
            const uint32_t r1 = r0 + 4;
            TT_SFPLOAD(p_sfpu::LREG0, mode, ADDR_MOD_7, r0);
            TT_SFPLOAD(p_sfpu::LREG1, mode, ADDR_MOD_7, r0 + 2);
            TT_SFPLOAD(p_sfpu::LREG2, mode, ADDR_MOD_7, r0 + ROWS_PER_FACE);
            TT_SFPLOAD(p_sfpu::LREG3, mode, ADDR_MOD_7, r0 + ROWS_PER_FACE + 2);
            TT_SFPLOAD(p_sfpu::LREG4, mode, ADDR_MOD_7, r1);
            TT_SFPLOAD(p_sfpu::LREG5, mode, ADDR_MOD_7, r1 + 2);
            TT_SFPLOAD(p_sfpu::LREG6, mode, ADDR_MOD_7, r1 + ROWS_PER_FACE);
            TT_SFPLOAD(p_sfpu::LREG7, mode, ADDR_MOD_7, r1 + ROWS_PER_FACE + 2);
            TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPABS_MOD1_FLOAT);
            TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPABS_MOD1_FLOAT);
            TTI_SFPABS(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPABS_MOD1_FLOAT);
            TTI_SFPABS(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPABS_MOD1_FLOAT);
            TTI_SFPABS(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPABS_MOD1_FLOAT);
            TTI_SFPABS(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPABS_MOD1_FLOAT);
            TTI_SFPABS(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPABS_MOD1_FLOAT);
            TTI_SFPABS(0, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPABS_MOD1_FLOAT);
            // Left / right faces and the two column halves of each face (same folds as sfpu_reduce's row MAX).
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
            TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
            horizontal_reduce_max();  // LREG0 / LREG4: row max of |x| in every column
            if constexpr (kPow2) {
                load_const_lreg(p_sfpu::LREG1, clamp_min_bits);
                load_const_lreg(p_sfpu::LREG7, clamp_min_bits);
                load_const_lreg(p_sfpu::LREG2, inv_448_bits);
                load_const_lreg(p_sfpu::LREG3, 0x007FFFFFu);
                load_const_lreg(p_sfpu::LREG5, 0x7F800000u);
                load_const_lreg(p_sfpu::LREG6, 254u << 23);
                pow2_inv_scale_lreg(p_sfpu::LREG0, p_sfpu::LREG1);
                pow2_inv_scale_lreg(p_sfpu::LREG4, p_sfpu::LREG7);
            }
            TT_SFPSTORE(p_sfpu::LREG0, mode, ADDR_MOD_7, r0);
            TT_SFPSTORE(p_sfpu::LREG4, mode, ADDR_MOD_7, r1);
        }
    }
}

}  // namespace ckernel::sfpu

inline void amax_pow2_inv_scale_tile(uint32_t idst, uint32_t clamp_min_bits, uint32_t inv_448_bits) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_amax_pow2_inv_scale,
        idst,
        VectorMode::RC_custom,
        clamp_min_bits,
        inv_448_bits);
}

inline void amax_rows_tile_call(uint32_t idst) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_amax_rows, idst, VectorMode::RC_custom, 0u, 0u);
}

inline void fold4_pow2_inv_scale_tile(uint32_t idst, uint32_t clamp_min_bits, uint32_t inv_448_bits) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_fold4_pow2_inv_scale,
        idst,
        VectorMode::RC_custom,
        clamp_min_bits,
        inv_448_bits);
}

inline void round_to_e4m3_rne_tile(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_round_to_e4m3_rne, (8 /* ITERATIONS */), idst, VectorMode::RC);
}

// Column 0 lives in faces 0 and 2, so VectorMode::C (the left faces) covers it at half the work.
inline void pow2_inv_scale_tile(uint32_t idst, uint32_t clamp_min_bits, uint32_t inv_448_bits) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_pow2_inv_scale,
        (8 /* ITERATIONS */),
        idst,
        VectorMode::C,
        clamp_min_bits,
        inv_448_bits);
}
#else
inline void round_to_e4m3_rne_tile(uint32_t) {}
inline void pow2_inv_scale_tile(uint32_t, uint32_t, uint32_t) {}
inline void amax_pow2_inv_scale_tile(uint32_t, uint32_t, uint32_t) {}
inline void amax_rows_tile_call(uint32_t) {}
inline void fold4_pow2_inv_scale_tile(uint32_t, uint32_t, uint32_t) {}
#endif
