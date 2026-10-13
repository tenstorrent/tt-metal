// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// silu(gate) * up in one SFPU pass over a gate / up tile pair in DST, run from the PACK thread so it overlaps
// the math thread's next subblock. The fp32 path uses silu_tile's accurate fp32 sigmoid; the bf16 path keeps
// silu_tile + mul_binary_tile's bf16 sigmoid and roundings.

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_agmm_swiglu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            result = up * (gate * _sfpu_sigmoid_<true>(gate));
        } else {
            sfpi::vFloat silu =
                sfpi::convert<sfpi::vFloat16b>(gate * _sfpu_sigmoid_<false>(gate), sfpi::RoundMode::Nearest);
            result = sfpi::convert<sfpi::vFloat16b>(silu * up, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

// _sfpu_sigmoid_'s reciprocal seed lives in vConstFloatPrgm0..2, which the binary SFPU init path does not program.
inline void agmm_swiglu_init() { sigmoid_init</*APPROXIMATION_MODE=*/false>(); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void agmm_swiglu_tile_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::agmm_swiglu_init);
}

inline void agmm_swiglu_tile(uint32_t gate_tile, uint32_t up_tile, uint32_t out_tile) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_agmm_swiglu,
        (DST_ACCUM_MODE, 8 /* ITERATIONS */),
        gate_tile,
        up_tile,
        out_tile,
        VectorMode::RC);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
