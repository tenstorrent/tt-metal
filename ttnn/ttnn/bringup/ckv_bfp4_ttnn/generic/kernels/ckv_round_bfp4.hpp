// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Round a DEST tile onto the bfloat4_b (MBITS 3) or bfloat8_b (MBITS 7) grid with tt-metal's host packer rule
// (convert_u32_to_bfp: per 16-value group the max exponent E, each value's 24-bit significand shifted by E - e, rounded
// to nearest even to MBITS bits, clamped at 2^MBITS - 1), so that the packer's own conversion (bfp4: round half away
// to bfp8, then drop 4 bits, truncation in effect; bfp8: half away, models/demos/deepseek_v41_flash_d_p/tests/
// test_bfp_packer_rounding.py) only copies bits. A value's group is
// its face row's 16 columns: an SFPU load holds 4 rows x the 8 even (or odd) columns, so the group max is the max of
// the even and odd loads, then across the 8 lanes of each row (rotate by 4, 2, 1).
// Exact against the host packer for groups whose max is >= 2^(MBITS - 127) (smaller ones come out 0).

#ifdef TRISC_MATH

#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vInt ckv_rot(sfpi::vInt v, int n) {
    for (int i = 0; i < n; i++) {
        v = sfpi::vInt(sfpi::subvec_shflror1(v));
    }
    return v;
}

sfpi_inline sfpi::vInt ckv_imax(sfpi::vInt a, sfpi::vInt b) {
    v_if(b > a) { a = b; }
    v_endif;
    return a;
}

// x rounded to q 2^(E - 127 - (MBITS - 1)) (q = host rule), sign kept, 0 as +0. The SFPU adder rounds to nearest
// even (tt-llk _round_even_), so |x| + M - M with M = 2^23 step (step = the group's LSB) is |x| rounded to a multiple
// of step; then clamped at (2^MBITS - 1) step (the host clamps q). Equal to the host rule for bf16 inputs: its
// alignment shift (at most MBITS + 1 bits) drops only zero bits of an 8-bit significand. Groups whose max is below
// 2^(MBITS - 127) come out 0; maxima above 2^(232 - 127) are out of range.
template <int MBITS>
sfpi_inline sfpi::vFloat ckv_round_one(sfpi::vFloat x, sfpi::vInt e_max) {
    constexpr float QMAX_FRAC = 2.0f - 1.0f / (1 << (MBITS - 1));  // (2^MBITS - 1) step / 2^(E - 127)
    const sfpi::vFloat m = sfpi::setexp(sfpi::vFloat(1.0f), e_max + (23 - (MBITS - 1)));
    sfpi::vFloat a = sfpi::setsgn(x, 0);
    a = a + m;
    a = a - m;
    const sfpi::vFloat lim = sfpi::setexp(sfpi::vFloat(QMAX_FRAC), e_max);
    v_if(a > lim) { a = lim; }
    v_endif;
    sfpi::vFloat r = sfpi::copysgn(a, x);
    v_if(a == 0.0f) { r = 0.0f; }
    v_endif;
    v_if(e_max < MBITS) { r = 0.0f; }
    v_endif;
    return r;
}

// One face (called per face by _llk_math_eltwise_unary_sfpu_params_ with VectorMode::RC).
template <int MBITS>
inline void ckv_round_bfp_face() {
#pragma GCC unroll 4
    for (int g = 0; g < 4; g++) {  // 4-row groups of the face
        sfpi::vFloat ev = sfpi::dst_reg[0];
        sfpi::vFloat od = sfpi::dst_reg[1];
        sfpi::vInt m =
            ckv_imax(sfpi::exexp(ev, sfpi::ExponentMode::Biased), sfpi::exexp(od, sfpi::ExponentMode::Biased));
        m = ckv_imax(m, ckv_rot(m, 4));
        m = ckv_imax(m, ckv_rot(m, 2));
        m = ckv_imax(m, ckv_rot(m, 1));
        sfpi::dst_reg[0] = ckv_round_one<MBITS>(ev, m);
        sfpi::dst_reg[1] = ckv_round_one<MBITS>(od, m);
        sfpi::dst_reg += 2;
    }
}

inline void ckv_round_bfp4_face() { ckv_round_bfp_face<3>(); }

}  // namespace ckernel::sfpu

#endif  // TRISC_MATH
