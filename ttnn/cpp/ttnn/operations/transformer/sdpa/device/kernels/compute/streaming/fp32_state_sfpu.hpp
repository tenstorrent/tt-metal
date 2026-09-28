// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {

// Frozen C/D state arithmetic. The new chunk is added by the L1 packer AFTER
// this FP32 multiplication, not fused into a SFPU MAD. Keep that rounding point.
template <int pairs>
inline void sdpa_state_rescale() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat correction = sfpi::dst_reg[32 * pairs];
#pragma GCC unroll 2
        for (int p = 0; p < pairs; ++p) {
            sfpi::vFloat old = sfpi::dst_reg[32 * p];
            sfpi::dst_reg[32 * p] = old * correction;
        }
        sfpi::dst_reg++;
    }
}

inline void sdpa_state_rescale_first_column() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 4; ++i) {
        sfpi::vFloat old = sfpi::dst_reg[0];
        sfpi::vFloat correction = sfpi::dst_reg[32];
        sfpi::dst_reg[0] = old * correction;
        sfpi::dst_reg += 2;
    }
}

inline void sdpa_state_normalize() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat out = sfpi::dst_reg[0];
        sfpi::vFloat inv = sfpi::dst_reg[32];
        sfpi::dst_reg[0] = out * inv;
        sfpi::dst_reg++;
    }
}

inline void sdpa_state_reciprocal() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    sfpi::vConstFloatPrgm0 = 2.0f;
    for (int d = 0; d < 4; ++d) {
        sfpi::dst_reg[0] = sfpu_reciprocal_iter<2>(sfpi::dst_reg[0]);
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
#endif
