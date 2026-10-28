// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Round a DEST tile onto the bfloat4_b grid with tt-metal's host packer rule (convert_u32_to_bfp: per 16-value
// group the max exponent E, each value's 24-bit significand shifted by E - e, rounded to nearest even to 3 bits,
// clamped at 7), so that the packer's own bfp4 conversion (round half away to bfp8, then drop 4 bits: truncation in
// effect, models/demos/deepseek_v41_flash_d_p/tests/test_bfp_packer_rounding.py) only copies bits. A value's group is
// its face row's 16 columns: an SFPU load holds 4 rows x the 8 even (or odd) columns, so the group max is the max of
// the even and odd loads, then across the 8 lanes of each row (rotate by 4, 2, 1).
// Exact against the host packer for groups whose max is >= 2^-124 (smaller ones come out 0).

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

// x rounded to q 2^(E - 129) (q = host rule), sign kept, 0 as +0. Round to nearest even in one add:
// (mm + 2^20 - 1 + lsb) >> 21. A shift past 4 already gives q = 0, so diff is capped there.
sfpi_inline sfpi::vFloat ckv_round_one(sfpi::vFloat x, sfpi::vInt e_max) {
    sfpi::vInt u = sfpi::as<sfpi::vInt>(x);
    sfpi::vInt ex = sfpi::exexp(x, sfpi::ExponentMode::Biased);
    sfpi::vInt diff = e_max - ex;
    v_if(diff > 4) { diff = 4; }
    v_endif;
    sfpi::vUInt man = sfpi::as<sfpi::vUInt>((u & 0x7FFFFF) | 0x800000);
    v_if(ex == 0) { man = 0; }
    v_endif;
    sfpi::vUInt mm = man >> sfpi::as<sfpi::vUInt>(diff);
    sfpi::vInt q = sfpi::as<sfpi::vInt>((mm + 0xFFFFF + ((mm >> 21) & 1)) >> 21);
    v_if(q > 7) { q = 7; }
    v_endif;
    sfpi::vFloat r = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(q), sfpi::RoundMode::Nearest);
    r = r * sfpi::setexp(sfpi::vFloat(1.0f), e_max - 2);
    v_if(x < 0.0f) { r = -r; }
    v_endif;
    v_if(q == 0) { r = 0.0f; }
    v_endif;
    v_if(e_max < 3) { r = 0.0f; }
    v_endif;
    return r;
}

// One face (called per face by _llk_math_eltwise_unary_sfpu_params_ with VectorMode::RC).
inline void ckv_round_bfp4_face() {
#pragma GCC unroll 4
    for (int g = 0; g < 4; g++) {  // 4-row groups of the face
        sfpi::vFloat ev = sfpi::dst_reg[0];
        sfpi::vFloat od = sfpi::dst_reg[1];
        sfpi::vInt m =
            ckv_imax(sfpi::exexp(ev, sfpi::ExponentMode::Biased), sfpi::exexp(od, sfpi::ExponentMode::Biased));
        m = ckv_imax(m, ckv_rot(m, 4));
        m = ckv_imax(m, ckv_rot(m, 2));
        m = ckv_imax(m, ckv_rot(m, 1));
        sfpi::dst_reg[0] = ckv_round_one(ev, m);
        sfpi::dst_reg[1] = ckv_round_one(od, m);
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu

#endif  // TRISC_MATH
