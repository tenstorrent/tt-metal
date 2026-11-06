// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post — fused SFPU mix (Refinement 5): the mix / y chains as a few multi-operand SFPU passes
// over DEST, fed by "packed" coefficient tiles, instead of one broadcast unpack + one copy + one
// SFPU op per term.
//
// Packed coefficient tile (Blackhole lane mapping, see mhc_post_hc_sfpu.hpp DENSE: SFPU vector k of
// a tile = dest rows 4*(k/2)..+3, the 8 columns of parity k%2; vectors 0..31 walk faces 0..3).
// A coefficient is a per-token (per-row) value, constant along a row. The 4 vectors k0 + {0, 1, 8, 9}
// (k0 = 16h + 2rg: parity 0/1 x face column 0/1) cover the same 4 rows, so a packed tile holds 4
// coefficients, coefficient s in "quadrant" Q[s] = {0, 1, 8, 9}[s] of every row group: the data vector
// in quadrant e and the coefficient in quadrant s are at constant offsets from k0, and the SFPU walks
// the 8 row groups with a moving dst base (small loops, all offsets immediate). CoefCompact builds a
// packed tile in DEST from four column-broadcast tiles (D0..D3); the mix then CopyTiles it (fp32,
// UnpackToDestFp32: exact) once per window.
//
// Mix window (one output tile X'_{j,c}, half-sync fp32 DEST = 4 slots), arithmetic order and
// rounding identical to the per-term chain (MulBinary then Addcmul(1.0) = SFPMAD, fp32 DEST):
//   D3 <- A_j = {post_j, comb_0j, comb_1j, comb_2j} (packed)   D0 <- F   D1 <- X_0   D2 <- X_1
//   MixPass1:  D0 = post_j*F ; D0 = comb_0j*X_0 + D0 ; D0 = comb_1j*X_1 + D0
//   D1 <- X_2;  MixPass2:  D0 = comb_2j*X_2 + D0
//   D3 <- bcast(comb_3j)  [D2 <- ss_run]  D1 <- X_3
//   MixPass3:  D0 = comb_3j*X_3 + D0  [ss: D2 = D0*D0 + D2]
// y window (collapse, y_c):  D3 <- P = {P_0..P_3} (packed)  D0 <- X'_0  D1 <- X'_1
//   YPass1: D0 = P_0*X'_0 ; D0 = P_1*X'_1 + D0 ;   D1 <- X'_2  D2 <- X'_3
//   YPass2: D0 = P_2*X'_2 + D0 ; D0 = P_3*X'_3 + D0

#pragma once

#include <cstdint>

#if defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_UNPACK)
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#endif

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

namespace mhc_post_mix {

constexpr uint32_t COEFS_PER_PACKED = 4;  // coefficients per packed tile
constexpr uint32_t VECS = 32;             // SFPU vectors per DEST tile

}  // namespace mhc_post_mix

#ifdef TRISC_MATH
namespace ckernel::sfpu {

namespace mhc_mix_detail {

using mhc_post_mix::VECS;

// Quadrant offsets: the 4 vectors k0 + Q[e] (k0 = 16h + 2rg) cover the same 4 rows (parity 0/1 x
// face column 0/1). Data vector e of a row group sits at k0 + Q[e]; packed coefficient s at k0 + Q[s].
constexpr uint32_t Q0 = 0, Q1 = 1, Q2 = 8, Q3 = 9;

// Walk the 8 row groups of a tile (dst base at k0): fn() sees every operand at a constant offset.
template <class Fn>
sfpi_inline void for_each_row_group(Fn fn) {
#pragma GCC unroll 0
    for (uint32_t h = 0; h < 2; ++h) {
#pragma GCC unroll 4
        for (uint32_t rg = 0; rg < 4; ++rg) {
            fn();
            sfpi::dst_reg += 2;
        }
        // += 8 vectors: the increment immediate is a signed 4-bit row count (2 rows per vector)
#pragma GCC unroll 4
        for (uint32_t i = 0; i < 4; ++i) {
            sfpi::dst_reg += 2;
        }
    }
}

}  // namespace mhc_mix_detail

// D0 <- packed {D0, D1, D2, D3} (column-broadcast coefficient tiles): coefficient s keeps quadrant
// Q[s] of its own broadcast tile (every quadrant of a broadcast tile holds the same rows), so
// quadrant Q0 of D0 is already in place.
inline void mhc_coef_compact() {
    using namespace mhc_mix_detail;
    for_each_row_group([] {
        const sfpi::vFloat a = sfpi::dst_reg[1 * VECS + Q1];
        const sfpi::vFloat b = sfpi::dst_reg[2 * VECS + Q2];
        const sfpi::vFloat c = sfpi::dst_reg[3 * VECS + Q3];
        sfpi::dst_reg[Q1] = a;
        sfpi::dst_reg[Q2] = b;
        sfpi::dst_reg[Q3] = c;
    });
}

// The passes process two quadrants at a time, their dependent MUL -> MAD -> MAD chains interleaved
// (as the stock addcmul does) so the SFPU pipeline does not stall on each result.

// Pass 1 on quadrants QA, QB: D0 = cp*D0 ; D0 = c0*D1 + D0 ; D0 = c1*D2 + D0
#define MHC_MIX_P1(QA, QB)                                  \
    {                                                       \
        sfpi::vFloat a = cp * sfpi::dst_reg[0 * VECS + QA]; \
        sfpi::vFloat b = cp * sfpi::dst_reg[0 * VECS + QB]; \
        a = c0 * sfpi::dst_reg[1 * VECS + QA] + a;          \
        b = c0 * sfpi::dst_reg[1 * VECS + QB] + b;          \
        a = c1 * sfpi::dst_reg[2 * VECS + QA] + a;          \
        b = c1 * sfpi::dst_reg[2 * VECS + QB] + b;          \
        sfpi::dst_reg[0 * VECS + QA] = a;                   \
        sfpi::dst_reg[0 * VECS + QB] = b;                   \
    }

// D0 = A0*D0 ; D0 = A1*D1 + D0 ; D0 = A2*D2 + D0   (A* = packed coefficients 0, 1, 2 of D3)
inline void mhc_mix_pass1() {
    using namespace mhc_mix_detail;
    for_each_row_group([] {
        const sfpi::vFloat cp = sfpi::dst_reg[3 * VECS + Q0];
        const sfpi::vFloat c0 = sfpi::dst_reg[3 * VECS + Q1];
        const sfpi::vFloat c1 = sfpi::dst_reg[3 * VECS + Q2];
        MHC_MIX_P1(Q0, Q1)
        MHC_MIX_P1(Q2, Q3)
    });
}
#undef MHC_MIX_P1

// D0 = A3*D1 + D0   (A3 = packed coefficient 3 of D3)
inline void mhc_mix_pass2() {
    using namespace mhc_mix_detail;
    for_each_row_group([] {
        const sfpi::vFloat c2 = sfpi::dst_reg[3 * VECS + Q3];
        const sfpi::vFloat a = c2 * sfpi::dst_reg[1 * VECS + Q0] + sfpi::dst_reg[Q0];
        const sfpi::vFloat b = c2 * sfpi::dst_reg[1 * VECS + Q1] + sfpi::dst_reg[Q1];
        const sfpi::vFloat c = c2 * sfpi::dst_reg[1 * VECS + Q2] + sfpi::dst_reg[Q2];
        const sfpi::vFloat d = c2 * sfpi::dst_reg[1 * VECS + Q3] + sfpi::dst_reg[Q3];
        sfpi::dst_reg[Q0] = a;
        sfpi::dst_reg[Q1] = b;
        sfpi::dst_reg[Q2] = c;
        sfpi::dst_reg[Q3] = d;
    });
}

// D0 = D3*D1 + D0 (D3 = column-broadcast coefficient, lane-aligned with the data) [; D2 = D0*D0 + D2]
template <bool SS>
inline void mhc_mix_pass3() {
    using mhc_post_mix::VECS;
#pragma GCC unroll 0
    for (uint32_t k = 0; k < VECS; k += 2) {
        const sfpi::vFloat a = sfpi::dst_reg[3 * VECS] * sfpi::dst_reg[1 * VECS] + sfpi::dst_reg[0];
        const sfpi::vFloat b = sfpi::dst_reg[3 * VECS + 1] * sfpi::dst_reg[1 * VECS + 1] + sfpi::dst_reg[1];
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg[1] = b;
        if constexpr (SS) {
            const sfpi::vFloat sa = a * a + sfpi::dst_reg[2 * VECS];
            const sfpi::vFloat sb = b * b + sfpi::dst_reg[2 * VECS + 1];
            sfpi::dst_reg[2 * VECS] = sa;
            sfpi::dst_reg[2 * VECS + 1] = sb;
        }
        sfpi::dst_reg += 2;
    }
}

// y pass 1 on quadrants QA, QB: D0 = P0*D0 ; D0 = P1*D1 + D0
#define MHC_Y_P1(QA, QB)                                    \
    {                                                       \
        sfpi::vFloat a = p0 * sfpi::dst_reg[0 * VECS + QA]; \
        sfpi::vFloat b = p0 * sfpi::dst_reg[0 * VECS + QB]; \
        a = p1 * sfpi::dst_reg[1 * VECS + QA] + a;          \
        b = p1 * sfpi::dst_reg[1 * VECS + QB] + b;          \
        sfpi::dst_reg[0 * VECS + QA] = a;                   \
        sfpi::dst_reg[0 * VECS + QB] = b;                   \
    }
// D0 = P0*D0 ; D0 = P1*D1 + D0   (P* = packed coefficients of D3)
inline void mhc_y_pass1() {
    using namespace mhc_mix_detail;
    for_each_row_group([] {
        const sfpi::vFloat p0 = sfpi::dst_reg[3 * VECS + Q0];
        const sfpi::vFloat p1 = sfpi::dst_reg[3 * VECS + Q1];
        MHC_Y_P1(Q0, Q1)
        MHC_Y_P1(Q2, Q3)
    });
}
#undef MHC_Y_P1

// y pass 2 on quadrants QA, QB: D0 = P2*D1 + D0 ; D0 = P3*D2 + D0
#define MHC_Y_P2(QA, QB)                                                        \
    {                                                                           \
        sfpi::vFloat a = p2 * sfpi::dst_reg[1 * VECS + QA] + sfpi::dst_reg[QA]; \
        sfpi::vFloat b = p2 * sfpi::dst_reg[1 * VECS + QB] + sfpi::dst_reg[QB]; \
        a = p3 * sfpi::dst_reg[2 * VECS + QA] + a;                              \
        b = p3 * sfpi::dst_reg[2 * VECS + QB] + b;                              \
        sfpi::dst_reg[QA] = a;                                                  \
        sfpi::dst_reg[QB] = b;                                                  \
    }
// D0 = P2*D1 + D0 ; D0 = P3*D2 + D0
inline void mhc_y_pass2() {
    using namespace mhc_mix_detail;
    for_each_row_group([] {
        const sfpi::vFloat p2 = sfpi::dst_reg[3 * VECS + Q2];
        const sfpi::vFloat p3 = sfpi::dst_reg[3 * VECS + Q3];
        MHC_Y_P2(Q0, Q1)
        MHC_Y_P2(Q2, Q3)
    });
}
#undef MHC_Y_P2

}  // namespace ckernel::sfpu
#endif

#if defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_UNPACK)
namespace mhc_post_mix {

// eltwise_chain element running one of the programs above over DEST slots D0..D3 (lane width 4;
// the chain owns DEST sync, CB lifecycle and reconfig).
template <class Derived>
struct SlotProgram : compute_kernel_lib::UnaryOp<Derived, compute_kernel_lib::Dst::D0> {
    static constexpr uint32_t lane_width = 4;
    static ALWI void init() {
        MATH((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::unused, DST_ACCUM_MODE>()));
    }
};

#define MHC_POST_SLOT_PROGRAM(NAME, FN)                                                                            \
    struct NAME : SlotProgram<NAME> {                                                                              \
        static ALWI void exec_impl(uint32_t slot_offset) {                                                         \
            MATH((_llk_math_eltwise_unary_sfpu_params_(::ckernel::sfpu::FN, slot_offset, VectorMode::RC_custom))); \
        }                                                                                                          \
    };

MHC_POST_SLOT_PROGRAM(CoefCompact, mhc_coef_compact)
MHC_POST_SLOT_PROGRAM(MixPass1, mhc_mix_pass1)
MHC_POST_SLOT_PROGRAM(MixPass2, mhc_mix_pass2)
MHC_POST_SLOT_PROGRAM(MixPass3, mhc_mix_pass3<false>)
MHC_POST_SLOT_PROGRAM(MixPass3Ss, mhc_mix_pass3<true>)
MHC_POST_SLOT_PROGRAM(YPass1, mhc_y_pass1)
MHC_POST_SLOT_PROGRAM(YPass2, mhc_y_pass2)
#undef MHC_POST_SLOT_PROGRAM

}  // namespace mhc_post_mix
#endif
