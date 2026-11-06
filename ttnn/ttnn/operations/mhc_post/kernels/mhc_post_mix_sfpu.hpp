// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post — fused SFPU mix (Refinement 5): the mix / y chains as a few multi-operand SFPU passes
// over DEST, fed by "packed" coefficient tiles, instead of one broadcast unpack + one copy + one
// SFPU op per term.
//
// Packed coefficient tile (Blackhole lane mapping, see mhc_post_hc_sfpu.hpp DENSE: SFPU vector k of
// a tile = dest rows 4*(k/2)..+3, the 8 columns of parity k%2; vectors 0..31 walk faces 0..3).
// A coefficient is a per-token (per-row) value, constant along a row, so every data vector that
// covers the same 4 rows (both parities, both face columns: k, k^1, k+8, k^1+8) uses the same
// coefficient vector. One coefficient for all 32 tokens is therefore 8 vectors, and a packed tile
// holds 4 coefficients: coefficient s of the tile at vectors 8*s + g, g = 4*(k/16) + (k%8)/2 the
// row group of data vector k. CoefCompact builds a packed tile in DEST from four column-broadcast
// tiles (D0..D3); the mix then CopyTiles it (fp32, UnpackToDestFp32: exact) once per window.
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
constexpr uint32_t GROUPS = 8;            // row groups (4 tokens each) per tile

}  // namespace mhc_post_mix

#ifdef TRISC_MATH
namespace ckernel::sfpu {

namespace mhc_mix_detail {

using mhc_post_mix::GROUPS;
using mhc_post_mix::VECS;

// The 4 data vectors of row group g (= 4*h + rg): rows 16h + 4rg..+3, parities 0/1, face columns 0/1.
constexpr uint32_t data_vec(uint32_t g, uint32_t e) { return 16 * (g / 4) + 2 * (g % 4) + (e & 1) + 8 * (e >> 1); }
// Packed coefficient s, row group g, of the tile in slot `slot`.
constexpr uint32_t coef_vec(uint32_t slot, uint32_t s, uint32_t g) { return slot * VECS + 8 * s + g; }
// The column-broadcast tile's vector for row group g (face 2h, row group rg, even parity).
constexpr uint32_t bcast_vec(uint32_t slot, uint32_t g) { return slot * VECS + 16 * (g / 4) + 2 * (g % 4); }

}  // namespace mhc_mix_detail

// D0 <- packed {D0, D1, D2, D3} (column-broadcast coefficient tiles). In place: every destination
// vector 8s + g of D0 is written after its own source reads (sources of D0 are vectors >= g).
inline void mhc_coef_compact() {
    using namespace mhc_mix_detail;
#pragma GCC unroll 4
    for (uint32_t s = 0; s < mhc_post_mix::COEFS_PER_PACKED; ++s) {
#pragma GCC unroll 8
        for (uint32_t g = 0; g < GROUPS; ++g) {
            const sfpi::vFloat v = sfpi::dst_reg[bcast_vec(s, g)];
            sfpi::dst_reg[coef_vec(0, s, g)] = v;
        }
    }
}

// D0 = A1*D0 ; D0 = A2*D1 + D0 ; D0 = A3*D2 + D0   (A* = packed coefficients 0, 1, 2 of D3)
inline void mhc_mix_pass1() {
    using namespace mhc_mix_detail;
#pragma GCC unroll 8
    for (uint32_t g = 0; g < GROUPS; ++g) {
        const sfpi::vFloat cp = sfpi::dst_reg[coef_vec(3, 0, g)];
        const sfpi::vFloat c0 = sfpi::dst_reg[coef_vec(3, 1, g)];
        const sfpi::vFloat c1 = sfpi::dst_reg[coef_vec(3, 2, g)];
#pragma GCC unroll 4
        for (uint32_t e = 0; e < 4; ++e) {
            const uint32_t k = data_vec(g, e);
            sfpi::vFloat acc = cp * sfpi::dst_reg[0 * VECS + k];
            acc = c0 * sfpi::dst_reg[1 * VECS + k] + acc;
            acc = c1 * sfpi::dst_reg[2 * VECS + k] + acc;
            sfpi::dst_reg[0 * VECS + k] = acc;
        }
    }
}

// D0 = A4*D1 + D0   (A4 = packed coefficient 3 of D3)
inline void mhc_mix_pass2() {
    using namespace mhc_mix_detail;
#pragma GCC unroll 8
    for (uint32_t g = 0; g < GROUPS; ++g) {
        const sfpi::vFloat c2 = sfpi::dst_reg[coef_vec(3, 3, g)];
#pragma GCC unroll 4
        for (uint32_t e = 0; e < 4; ++e) {
            const uint32_t k = data_vec(g, e);
            sfpi::dst_reg[0 * VECS + k] = c2 * sfpi::dst_reg[1 * VECS + k] + sfpi::dst_reg[0 * VECS + k];
        }
    }
}

// D0 = D3*D1 + D0 (D3 = column-broadcast coefficient, lane-aligned with the data) [; D2 = D0*D0 + D2]
template <bool SS>
inline void mhc_mix_pass3() {
    using mhc_post_mix::VECS;
#pragma GCC unroll 4
    for (uint32_t k = 0; k < VECS; ++k) {
        const sfpi::vFloat acc = sfpi::dst_reg[3 * VECS] * sfpi::dst_reg[1 * VECS] + sfpi::dst_reg[0];
        sfpi::dst_reg[0] = acc;
        if constexpr (SS) {
            sfpi::dst_reg[2 * VECS] = acc * acc + sfpi::dst_reg[2 * VECS];
        }
        sfpi::dst_reg++;
    }
}

// D0 = P0*D0 ; D0 = P1*D1 + D0   (P* = packed coefficients of D3)
inline void mhc_y_pass1() {
    using namespace mhc_mix_detail;
#pragma GCC unroll 8
    for (uint32_t g = 0; g < GROUPS; ++g) {
        const sfpi::vFloat p0 = sfpi::dst_reg[coef_vec(3, 0, g)];
        const sfpi::vFloat p1 = sfpi::dst_reg[coef_vec(3, 1, g)];
#pragma GCC unroll 4
        for (uint32_t e = 0; e < 4; ++e) {
            const uint32_t k = data_vec(g, e);
            sfpi::vFloat acc = p0 * sfpi::dst_reg[0 * VECS + k];
            acc = p1 * sfpi::dst_reg[1 * VECS + k] + acc;
            sfpi::dst_reg[0 * VECS + k] = acc;
        }
    }
}

// D0 = P2*D1 + D0 ; D0 = P3*D2 + D0
inline void mhc_y_pass2() {
    using namespace mhc_mix_detail;
#pragma GCC unroll 8
    for (uint32_t g = 0; g < GROUPS; ++g) {
        const sfpi::vFloat p2 = sfpi::dst_reg[coef_vec(3, 2, g)];
        const sfpi::vFloat p3 = sfpi::dst_reg[coef_vec(3, 3, g)];
#pragma GCC unroll 4
        for (uint32_t e = 0; e < 4; ++e) {
            const uint32_t k = data_vec(g, e);
            sfpi::vFloat acc = p2 * sfpi::dst_reg[1 * VECS + k] + sfpi::dst_reg[0 * VECS + k];
            acc = p3 * sfpi::dst_reg[2 * VECS + k] + acc;
            sfpi::dst_reg[0 * VECS + k] = acc;
        }
    }
}

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
