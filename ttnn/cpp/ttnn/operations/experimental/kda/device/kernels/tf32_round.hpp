// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// fp32 DEST values rounded to TF32 (10 mantissa bits, nearest-even) before they are packed into an fp32 CB that an FPU
// op reads: unpacking fp32 into SrcA / SrcB drops the low 13 mantissa bits (toward zero), a systematic bias of about
// -2^-11 ln 2 relative per operand; rounded first, that truncation drops only zeros. (Infinities stay infinite: the
// mask clears what the rounding added.)

#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {
inline void kda_round_tf32() {
#pragma GCC unroll 8
    for (int face_row = 0; face_row < 8; face_row++) {
        sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[0]));
        const sfpi::vUInt lsb = (bits >> 13) & 1;
        bits = (bits + 0xFFFu + lsb) & 0xFFFFE000u;
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(bits);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

ALWI void kda_round_tf32_init() { MATH((SFPU_UNARY_INIT(unused))); }
ALWI void kda_round_tf32_tile(uint32_t dst_index, VectorMode vector_mode = VectorMode::RC) {
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, kda_round_tf32, dst_index, vector_mode)));
}
