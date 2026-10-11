// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 1x32 fp8 quantize helper shared by per_token_cast_to_fp8 (FAST_1X32) and the routed expert's Q2 epilogue:
// from a row-max tile (column 0, one value per tile row), write the exact power-of-two 1/scale in place.
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

}  // namespace ckernel::sfpu

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
inline void pow2_inv_scale_tile(uint32_t, uint32_t, uint32_t) {}
#endif
