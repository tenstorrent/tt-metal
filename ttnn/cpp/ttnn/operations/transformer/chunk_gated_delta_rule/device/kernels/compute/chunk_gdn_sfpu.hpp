// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused SFPU passes of the chunk_gdn prep (math RISC only). Each replaces a chain of one-op full-tile SFPU passes
// (sub, mul, add, exp, rsqrt, negative: one init and 32 DST vectors each) by ONE pass that keeps the chain in
// LREGs and visits only the DST vectors whose results are read downstream. The per-element arithmetic is the
// same precise fp32 bodies the tile ops use (_sfpu_exp_fp32_accurate_, _calculate_sqrt_body_), in the same order.
//
// DST addressing (32-bit accumulation): a 32x32 tile is 32 sfpi DST vectors, faces f0..f3 at vectors 0, 8, 16,
// 24; vector 2r + 0 of a face holds its rows 4r..4r+3, even columns, vector 2r + 1 the odd columns. Column 0 of a
// tile is therefore the even vectors of faces 0 and 2. The face walk follows _llk_math_eltwise_sfpu_apply_vector_mode_:
// dst_reg++ within a face, then two 8-row SETRWCs to the next face's start.

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_sfpu_common.h"
#endif

namespace gdn_sfpu {

#ifdef TRISC_MATH
inline constexpr uint32_t kTileVecs = 32;  // sfpi DST vectors per 32x32 fp32 tile
inline constexpr uint32_t kFaceVecs = 8;   // per 16x16 face

inline void next_face() { _llk_math_eltwise_sfpu_inc_dst_face_addr_(); }

// decay_all's SFPU tail, column 0 only (every reader of these tiles broadcasts column 0):
//   DST[FAC] = exp(DST[FAC] - DST[DEC])    decayfac = exp(g_sum - decay)
//   DST[EXP] = exp(DST[DEC])               decay_exp
//   DST[DL]  = DST[FAC] * DST[EXP]         dl = exp(g_sum), only when WITH_DL
// Column 0 sits in lane 8j (j = row within the 4-row block) of the 8 even-column vectors of faces 0 and 2 (kCol0Vecs),
// four useful lanes per vector. The pass gathers those 32 values into ONE vector -- vector k's column-0 lanes go to
// lanes 8j + k, by seven subvec rotate-right-by-one steps with a lane-0 select after each -- so each exp body runs once
// instead of once per vector (2 bodies instead of 16). The results are scattered back by the inverse rotation: after m
// rotations lane 8j holds what vector (8 - m) % 8 needs. Each column-0 element goes through the same fp32 ops on the
// same operands as the per-vector form, so the column-0 values are bit-identical; the other lanes of the stored vectors
// receive other rows' results instead of their own columns' (never read, as before). The gathered decay and decayfac
// are stashed in odd-column vectors of FAC (columns never read) between the bodies, which need most of the LREGs.
inline constexpr uint32_t kCol0Vecs[8] = {0, 2, 4, 6, 16, 18, 20, 22};
inline constexpr uint32_t kStashDec = 1;  // odd-column vectors of FAC
inline constexpr uint32_t kStashFac = 3;

// Gathered vector -> column 0 of the 8 column-0 vectors of tile `base` (in sfpi vectors).
template <uint32_t BASE>
inline void scatter_col0(sfpi::vFloat g) {
    sfpi::dst_reg[BASE + kCol0Vecs[0]] = g;
#pragma GCC unroll 7
    for (uint32_t m = 1; m < 8; m++) {
        g = sfpi::subvec_shflror1(g);  // lane 8j <- lane 8j + 8 - m
        sfpi::dst_reg[BASE + kCol0Vecs[8 - m]] = g;
    }
}

template <uint32_t DEC, uint32_t FAC, uint32_t DL, uint32_t EXP, bool WITH_DL>
inline void decay_exps_body() {
    constexpr uint32_t dec0 = DEC * kTileVecs, fac0 = FAC * kTileVecs, dl0 = DL * kTileVecs, exp0 = EXP * kTileVecs;
    {
        // Gather: start from vector 7, rotate right by one and take the next vector's lane 8j into lane 8j.
        const sfpi::vInt col = sfpi::vConstTileId & static_cast<int>(0xE);  // 2 * (lane % 8)
        sfpi::vFloat d = sfpi::dst_reg[dec0 + kCol0Vecs[7]];
        sfpi::vFloat s = sfpi::dst_reg[fac0 + kCol0Vecs[7]];
#pragma GCC unroll 7
        for (int k = 6; k >= 0; k--) {
            d = sfpi::subvec_shflror1(d);
            s = sfpi::subvec_shflror1(s);
            v_if(col == 0) {
                d = sfpi::dst_reg[dec0 + kCol0Vecs[k]];
                s = sfpi::dst_reg[fac0 + kCol0Vecs[k]];
            }
            v_endif;
        }
        sfpi::dst_reg[fac0 + kStashDec] = d;
        sfpi::dst_reg[fac0 + kStashFac] = ckernel::sfpu::_sfpu_exp_fp32_accurate_(s - d);
    }
    {
        const sfpi::vFloat d = sfpi::dst_reg[fac0 + kStashDec];
        const sfpi::vFloat e = ckernel::sfpu::_sfpu_exp_fp32_accurate_(d);
        scatter_col0<exp0>(e);
        if constexpr (WITH_DL) {
            const sfpi::vFloat fac = sfpi::dst_reg[fac0 + kStashFac];
            const sfpi::vFloat e2 = sfpi::dst_reg[exp0 + kCol0Vecs[0]];  // e, as stored
            scatter_col0<dl0>(fac * e2);
        }
    }
    scatter_col0<fac0>(sfpi::dst_reg[fac0 + kStashFac]);
}

// L_mask = tril * exp(tril * (DST[DI] - DST[DJ])) -> DST[DI], with tril in DST[TRIL]. Face 1 (rows 0..15, columns
// 16..31) is strictly upper, so it is stored as 0 (what the chain gives there) without running it. With NEG, the
// column-0 vectors of DST[NEG] (faces 0 and 2) are sign-flipped in the same pass.
template <uint32_t DI, uint32_t DJ, uint32_t TRIL, uint32_t NEG, bool WITH_NEG>
inline void lmask_body() {
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        if (face == 1) {
#pragma GCC unroll 0
            for (uint32_t v = 0; v < kFaceVecs; v++) {
                sfpi::dst_reg[DI * kTileVecs] = 0.0f;
                sfpi::dst_reg++;
            }
        } else {
#pragma GCC unroll 0
            for (uint32_t v = 0; v < kFaceVecs; v++) {
                sfpi::vFloat x;
                {
                    const sfpi::vFloat m = sfpi::dst_reg[TRIL * kTileVecs];
                    const sfpi::vFloat di = sfpi::dst_reg[DI * kTileVecs];
                    const sfpi::vFloat dj = sfpi::dst_reg[DJ * kTileVecs];
                    x = (di - dj) * m;  // zero the upper triangle before the exp
                }
                x = ckernel::sfpu::_sfpu_exp_fp32_accurate_(x);
                {
                    const sfpi::vFloat m = sfpi::dst_reg[TRIL * kTileVecs];  // re-read: the exp needs the LREGs
                    sfpi::dst_reg[DI * kTileVecs] = x * m;
                }
                if constexpr (WITH_NEG) {
                    if (face != 3) {
                        const sfpi::vFloat n = sfpi::dst_reg[NEG * kTileVecs];
                        sfpi::dst_reg[NEG * kTileVecs] = -n;
                    }
                }
                sfpi::dst_reg++;
            }
        }
        next_face();
    }
}

// inv_rms_diag's SFPU chain: DST[G] = I * (rsqrt(G * I + eps) [* scale]) with I in DST[EYE], over the diagonal faces
// 0 and 3; the off-diagonal faces 1 and 2 are stored as 0 (what the chain gives there).
template <uint32_t G, uint32_t EYE, bool DO_SCALE>
inline void inv_rms_body(const uint32_t eps_bits, const uint32_t scale_bits) {
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        if (face == 1 || face == 2) {
#pragma GCC unroll 0
            for (uint32_t v = 0; v < kFaceVecs; v++) {
                sfpi::dst_reg[G * kTileVecs] = 0.0f;
                sfpi::dst_reg++;
            }
        } else {
#pragma GCC unroll 0
            for (uint32_t v = 0; v < kFaceVecs; v++) {
                // Constants and the identity are (re)loaded next to their use: the rsqrt body needs the LREGs.
                sfpi::vFloat x;
                {
                    const sfpi::vFloat id = sfpi::dst_reg[EYE * kTileVecs];
                    const sfpi::vFloat gv = sfpi::dst_reg[G * kTileVecs];
                    x = gv * id;
                }
                x = x + sfpi::vFloat(ckernel::sfpu::Converter::as_float(eps_bits));
                x = ckernel::sfpu::_calculate_sqrt_body_<false, true, false>(x);
                if constexpr (DO_SCALE) {
                    x = x * sfpi::vFloat(ckernel::sfpu::Converter::as_float(scale_bits));
                }
                {
                    const sfpi::vFloat id = sfpi::dst_reg[EYE * kTileVecs];
                    sfpi::dst_reg[G * kTileVecs] = x * id;
                }
                sfpi::dst_reg++;
            }
        }
        next_face();
    }
}

template <typename Body>
inline void run(Body&& body) {
    _llk_math_eltwise_sfpu_start_(0);  // tile indices are absolute (DST base 0)
    body();
    _llk_math_eltwise_sfpu_done_();
}
#endif

}  // namespace gdn_sfpu

// Thread-agnostic wrappers (no-ops off the math RISC). Kept out of line: one copy each in the program.
template <uint32_t DEC, uint32_t FAC, uint32_t DL, uint32_t EXP, bool WITH_DL>
[[gnu::noinline]] inline void gdn_sfpu_decay_exps() {
    MATH((gdn_sfpu::run([] { gdn_sfpu::decay_exps_body<DEC, FAC, DL, EXP, WITH_DL>(); })));
}

template <uint32_t DI, uint32_t DJ, uint32_t TRIL, uint32_t NEG, bool WITH_NEG>
[[gnu::noinline]] inline void gdn_sfpu_lmask() {
    MATH((gdn_sfpu::run([] { gdn_sfpu::lmask_body<DI, DJ, TRIL, NEG, WITH_NEG>(); })));
}

template <uint32_t G, uint32_t EYE, bool DO_SCALE>
[[gnu::noinline]] inline void gdn_sfpu_inv_rms(uint32_t eps_bits, uint32_t scale_bits) {
    MATH((gdn_sfpu::run([&] { gdn_sfpu::inv_rms_body<G, EYE, DO_SCALE>(eps_bits, scale_bits); })));
}
