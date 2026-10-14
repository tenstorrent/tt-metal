// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//=============================================================================
// SwiGLU input clamp for SFPU (runs on PACK thread), used before SiLU(gate) * up
//
//   gate = min(gate, limit)
//   up   = clamp(up, -limit, limit)
//
// This is the DeepSeek-V4 expert activation (swiglu_limit = 10): the reference Expert clamps both projections
// and then computes silu(gate) * up. Config::limit supplies the limit as a compile-time float.
//=============================================================================

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config>
inline void calculate_swiglu_clamp(const uint gate_tile_idx, const uint up_tile_idx, const uint /*out_tile_idx*/) {
    constexpr float limit = Config::limit;
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        v_if(gate > limit) { gate = limit; }
        v_endif;
        v_if(up > limit) { up = limit; }
        v_endif;
        v_if(up < -limit) { up = -limit; }
        v_endif;

        sfpi::dst_reg[gate_tile_idx * dst_tile_size] = gate;
        sfpi::dst_reg[up_tile_idx * dst_tile_size] = up;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

namespace ckernel {

template <bool is_fp32_dest_acc_en, class Config>
inline void llk_math_eltwise_binary_sfpu_swiglu_clamp(
    uint gate_tile, uint32_t up_tile, VectorMode vector_mode = VectorMode::RC) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_swiglu_clamp,
        (is_fp32_dest_acc_en, 8 /*ITERATIONS*/, Config),
        gate_tile,
        up_tile,
        up_tile,
        vector_mode);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
