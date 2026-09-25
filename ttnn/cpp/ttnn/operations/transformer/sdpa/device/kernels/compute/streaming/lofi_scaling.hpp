// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/bcast.h"

// Separate O(ND) recurrent/output scaling fidelity from the O(N^2 D) matmuls.
// LoFi SrcA only consumes five significant bits; applying it to recurrent O
// loses precision even when the broadcast multiplier is exactly one.
namespace ckernel {
ALWI void lofi_safe_mul_bcast_cols_init(uint32_t a, uint32_t b, uint32_t line = __builtin_LINE()) {
    state_configure(a, b, line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MathFidelity::HiFi2>(a, b)));
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(a, b)));
}
template <bool fp32 = DST_ACCUM_MODE>
ALWI void lofi_safe_mul_tiles_bcast_cols(uint32_t a, uint32_t b, uint32_t ai, uint32_t bi, uint32_t dst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::COL,
          fp32,
          MathFidelity::HiFi2,
          EltwiseBinaryReuseDestType::NONE>(a, b, dst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(a, b, ai, bi)));
}
}  // namespace ckernel
#define mul_bcast_cols_init lofi_safe_mul_bcast_cols_init
#define mul_tiles_bcast_cols lofi_safe_mul_tiles_bcast_cols
