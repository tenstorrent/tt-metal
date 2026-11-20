// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

// silu(gate * scale) * (up * scale) with exp(-gate) from a linear-mantissa 2^x (about 3% relative error) and the
// hardware reciprocal
template <int ITERATIONS = 8>
inline void calculate_swiglu(const uint gate_tile, const uint up_tile, const uint out_tile, const uint scale_tile) {
    constexpr uint dst_tile_rows = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat scale = sfpi::dst_reg[scale_tile * dst_tile_rows];
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile * dst_tile_rows] * scale;
        sfpi::vFloat up = sfpi::dst_reg[up_tile * dst_tile_rows] * scale;
        sfpi::vFloat exponent = sfpi::clamp(gate * -1.4426950216293334961f + 126.9570f, 0.0f, 255.0f);
        sfpi::vFloat exp_neg_gate = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(exponent));
        sfpi::dst_reg[out_tile * dst_tile_rows] = up * (gate * sfpi::approx_recip(1.0f + exp_neg_gate));
        sfpi::dst_reg++;
    }
}

inline void swiglu_init() {}

}  // namespace ckernel::sfpu

namespace ckernel {

inline void swiglu_tile_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::swiglu_init); }

inline void swiglu_tile(uint32_t gate_tile, uint32_t up_tile, uint32_t out_tile, uint32_t scale_tile) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_swiglu,
        (8 /* ITERATIONS */),
        gate_tile,
        up_tile,
        out_tile,
        VectorMode::RC,
        scale_tile);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
