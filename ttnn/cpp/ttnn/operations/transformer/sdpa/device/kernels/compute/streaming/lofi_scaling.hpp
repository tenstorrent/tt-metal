// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/bcast.h"

// Final output normalization O * (1/l), broadcast along columns.
//
// FAST (SDPA_RECIPE_LOFI) runs its compute config at LoFi, where the FPU multiplier consumes only
// the top 5 significant bits of SrcA. Its matmuls take inputs already rounded to that width, but O is not,
// so the O(N*D) normalization runs at HiFi2 (SrcA fully consumed) while the O(N^2*D) matmuls stay LoFi.
// Every other recipe uses the standard mul_bcast_cols at its compute-config fidelity.
namespace ckernel {
ALWI void recipe_output_scale_init(uint32_t out_cb, uint32_t scale_cb, uint32_t line = __builtin_LINE()) {
#ifdef SDPA_RECIPE_LOFI
    state_configure(out_cb, scale_cb, line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MathFidelity::HiFi2>(
        out_cb, scale_cb)));
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(out_cb, scale_cb)));
#else
    mul_bcast_cols_init(out_cb, scale_cb);
#endif
}

ALWI void recipe_output_scale_tile(uint32_t out_cb, uint32_t scale_cb, uint32_t out_tile, uint32_t scale_tile, uint32_t dst) {
#ifdef SDPA_RECIPE_LOFI
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::COL,
          DST_ACCUM_MODE,
          MathFidelity::HiFi2,
          EltwiseBinaryReuseDestType::NONE>(out_cb, scale_cb, dst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(out_cb, scale_cb, out_tile, scale_tile)));
#else
    mul_tiles_bcast_cols(out_cb, scale_cb, out_tile, scale_tile, dst);
#endif
}
}  // namespace ckernel
