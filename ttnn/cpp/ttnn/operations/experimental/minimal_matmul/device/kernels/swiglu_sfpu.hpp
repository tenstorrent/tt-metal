// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// silu(gate) * up in one SFPU pass over a gate / up tile pair in DST, drivable from the MATH or the PACK thread
// (minimal_matmul's fused SwiGLU runs it on the PACK thread, so it overlaps the math thread's next subblock). Same
// sigmoid and bf16 roundings as silu_tile followed by mul_binary_tile. The pattern follows
// ckernel_sfpu_clamped_silu_glu.h and moe_gpt's swiglu_sfpu.h, without their clamps / alpha / up + 1.

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_minimal_matmul_swiglu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat silu = gate * _sfpu_sigmoid_<is_fp32_dest_acc_en>(gate);
        if constexpr (!is_fp32_dest_acc_en) {
            silu = sfpi::convert<sfpi::vFloat16b>(silu, sfpi::RoundMode::Nearest);
        }
        sfpi::vFloat result = silu * up;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

// _sfpu_sigmoid_ takes its reciprocal from sfpu_reciprocal_iter, which needs vConstFloatPrgm0 = 2.0f; nothing on the
// binary SFPU init path programs it.
inline void minimal_matmul_swiglu_init() { sigmoid_init</*APPROXIMATION_MODE=*/false>(); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void llk_minimal_matmul_swiglu_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::minimal_matmul_swiglu_init);
}

inline void llk_minimal_matmul_swiglu(uint gate_tile, uint32_t up_tile, uint32_t out_tile) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_minimal_matmul_swiglu,
        (DST_ACCUM_MODE, 8 /* ITERATIONS */),
        gate_tile,
        up_tile,
        out_tile,
        VectorMode::RC);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
