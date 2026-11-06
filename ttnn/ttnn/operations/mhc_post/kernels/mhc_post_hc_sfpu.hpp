// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post — the hc step (DeepSeek-V4 hc_split_sinkhorn after the projection) as one custom fp32
// SFPU program over "hc lane" tiles, plus its eltwise_chain element (HcProgram).
//
// Lane layout (op_design.md → Dataflow "stats → hc lanes", the robust layout): quantity k
// (0..23 = the stats mix columns, 24 = the sum of squares) occupies dest rows 4·(k%16)..+3 of lane
// tile k/16 (all 16 columns) — i.e. SFPU vectors dst_reg[2k] and dst_reg[2k+1] counted from lane
// tile 0. Token t sits at rel row t/16, col t%16 of the 4-row block, and rel rows 2..3 duplicate
// rows 0..1. Every quantity block is therefore a whole number of SFPU loads, so every op below is
// lane-wise between quantities whatever the intra-load datum mapping is; both halves (vector 2k
// and 2k+1) are processed identically.
//
// Program per token (all fp32, lane-wise, exact transcendentals, no FPU operand):
//   r     = rsqrt(ss * (1/width) + norm_eps)                    (magic seed + 3 Newton steps)
//   pre   = sigmoid(a_pre  * (mix * r) + b) + hc_eps            (k = 0..n-1)
//   post  = 2 sigmoid(a_post * (mix * r) + b)                   (k = n..2n-1)
//   L     = a_res * (mix * r) + b, n x n row-major              (k = 2n..)
//   M     = softmax_rows(L) (row max subtracted) + hc_eps
//   M     = M / (col sums + hc_eps)
//   repeat iters - 1:  M = M / (row sums + hc_eps);  M = M / (col sums + hc_eps)
// Divisions are an exact reciprocal (approx + 2 Newton) times a multiply. Results overwrite the
// quantity slots 0..n(n+2)-1 in place; slot n(n+2) (ss) ends up holding r (don't-care).

#pragma once

#include <cstdint>

// Compute-only parts (the chain element and the SFPU program) are compiled for the TRISCs only;
// the writer includes this header for the lane layout constants.
#if defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_UNPACK)
#define MHC_POST_HC_COMPUTE 1
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#endif

#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace mhc_post_hc {

constexpr uint32_t N = 4;
constexpr uint32_t NG = N * (N + 2);  // 24 hc quantities
constexpr uint32_t SS_Q = NG;         // quantity index of the sum of squares
constexpr uint32_t LANE_TILES = 2;    // 25 quantities x 64 datums -> 2 tiles

// Runtime parameter block (compute RT args, fp32 bits unless noted).
constexpr uint32_t P_BASE = 0;            // b_0 .. b_23
constexpr uint32_t P_A_PRE = NG + 0;      // a_pre
constexpr uint32_t P_A_POST = NG + 1;     // a_post
constexpr uint32_t P_A_RES = NG + 2;      // a_res
constexpr uint32_t P_INV_WIDTH = NG + 3;  // 1 / hc_norm_width (host double -> fp32)
constexpr uint32_t P_NORM_EPS = NG + 4;   // hc_norm_eps
constexpr uint32_t P_HC_EPS = NG + 5;     // hc_eps
constexpr uint32_t P_ITERS = NG + 6;      // hc_sinkhorn_iters (integer)
constexpr uint32_t NUM_PARAMS = NG + 7;

// Writer-side lane addressing: fp32 word of (quantity k, token t) inside lane tile k / 16
// (the duplicate sits at + 32).
constexpr uint32_t lane_word(uint32_t k, uint32_t t) { return (k % 16) * 64 + t; }

}  // namespace mhc_post_hc

#ifdef TRISC_MATH
namespace ckernel::sfpu {

namespace mhc_hc_detail {

constexpr uint32_t q(uint32_t k) { return 2 * k; }  // SFPU vector index of quantity k (this half)

sfpi_inline sfpi::vFloat bits(uint32_t v) { return Converter::as_float(v); }

sfpi_inline sfpi::vFloat ld(uint32_t idx) { return sfpi::dst_reg[idx]; }

// 1 / x for finite x > 0: hardware approximation + 2 Newton steps (vConstFloatPrgm0 = 2.0).
sfpi_inline sfpi::vFloat recip(sfpi::vFloat x) { return sfpu_reciprocal_iter<2>(x); }

// 1 / sqrt(x) for finite x > 0: bit-trick seed + 3 Newton steps (~3e-11 before fp32 rounding).
sfpi_inline sfpi::vFloat rsqrt(sfpi::vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f3759df) - i);
    sfpi::vFloat hx = x * 0.5f;
#pragma GCC unroll 3
    for (int it = 0; it < 3; ++it) {
        y = y * (1.5f - hx * y * y);
    }
    return y;
}

// sigmoid(t) = 1 / (1 + exp(-t)), exact exp.
sfpi_inline sfpi::vFloat sigmoid(sfpi::vFloat t) {
    sfpi::vFloat e = _sfpu_exp_fp32_accurate_<false>(-t);
    return recip(e + 1.0f);
}

// M[j][k] at quantity 2n + j*n + k.
constexpr uint32_t m(uint32_t j, uint32_t k) { return q(2 * mhc_post_hc::N + j * mhc_post_hc::N + k); }

sfpi_inline void row_normalize(uint32_t hc_eps_bits) {
    constexpr uint32_t n = mhc_post_hc::N;
#pragma GCC unroll 4
    for (uint32_t j = 0; j < n; ++j) {
        sfpi::vFloat s = ld(m(j, 0)) + ld(m(j, 1));
        s = s + sfpi::dst_reg[m(j, 2)];
        s = s + sfpi::dst_reg[m(j, 3)];
        sfpi::vFloat inv = recip(s + bits(hc_eps_bits));
#pragma GCC unroll 4
        for (uint32_t k = 0; k < n; ++k) {
            sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] * inv;
        }
    }
}

sfpi_inline void col_normalize(uint32_t hc_eps_bits) {
    constexpr uint32_t n = mhc_post_hc::N;
#pragma GCC unroll 4
    for (uint32_t k = 0; k < n; ++k) {
        sfpi::vFloat s = ld(m(0, k)) + ld(m(1, k));
        s = s + sfpi::dst_reg[m(2, k)];
        s = s + sfpi::dst_reg[m(3, k)];
        sfpi::vFloat inv = recip(s + bits(hc_eps_bits));
#pragma GCC unroll 4
        for (uint32_t j = 0; j < n; ++j) {
            sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] * inv;
        }
    }
}

}  // namespace mhc_hc_detail

// The whole hc program over the lane tiles starting at the current dest base (RC_custom: called
// once). `p` = the mhc_post_hc parameter block.
inline void calculate_mhc_hc(const uint32_t* p) {
    using namespace mhc_hc_detail;
    constexpr uint32_t n = mhc_post_hc::N;
    sfpi::vConstFloatPrgm0 = 2.0f;  // Newton constant of sfpu_reciprocal_iter
    const uint32_t iters = p[mhc_post_hc::P_ITERS];
    const uint32_t hc_eps = p[mhc_post_hc::P_HC_EPS];

#pragma GCC unroll 0
    for (int half = 0; half < 2; ++half) {
        // r = rsqrt(ss / width + norm_eps), kept in the ss slot
        {
            sfpi::vFloat ss = sfpi::dst_reg[q(mhc_post_hc::SS_Q)];
            sfpi::vFloat x = ss * bits(p[mhc_post_hc::P_INV_WIDTH]) + bits(p[mhc_post_hc::P_NORM_EPS]);
            sfpi::dst_reg[q(mhc_post_hc::SS_Q)] = rsqrt(x);
        }
        // pre / post
#pragma GCC unroll 4
        for (uint32_t k = 0; k < n; ++k) {
            sfpi::vFloat z = ld(q(k)) * ld(q(mhc_post_hc::SS_Q));
            sfpi::vFloat t = z * bits(p[mhc_post_hc::P_A_PRE]) + bits(p[mhc_post_hc::P_BASE + k]);
            sfpi::dst_reg[q(k)] = sigmoid(t) + bits(hc_eps);
        }
#pragma GCC unroll 4
        for (uint32_t k = n; k < 2 * n; ++k) {
            sfpi::vFloat z = ld(q(k)) * ld(q(mhc_post_hc::SS_Q));
            sfpi::vFloat t = z * bits(p[mhc_post_hc::P_A_POST]) + bits(p[mhc_post_hc::P_BASE + k]);
            sfpi::vFloat s = sigmoid(t);
            sfpi::dst_reg[q(k)] = s + s;
        }
        // logits + row softmax (row max subtracted) + hc_eps (unrolled: dst_reg indices stay immediates)
#pragma GCC unroll 4
        for (uint32_t j = 0; j < n; ++j) {
            sfpi::vFloat mx = 0.0f;
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t qq = 2 * n + j * n + k;
                sfpi::vFloat z = ld(m(j, k)) * ld(q(mhc_post_hc::SS_Q));
                sfpi::vFloat l = z * bits(p[mhc_post_hc::P_A_RES]) + bits(p[mhc_post_hc::P_BASE + qq]);
                sfpi::dst_reg[m(j, k)] = l;
                if (k == 0) {
                    mx = l;
                } else {
                    v_if(l > mx) { mx = l; }
                    v_endif;
                }
            }
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] - mx;
            }
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                sfpi::dst_reg[m(j, k)] = _sfpu_exp_fp32_accurate_<false>(sfpi::dst_reg[m(j, k)]);
            }
            sfpi::vFloat s = ld(m(j, 0)) + ld(m(j, 1));
            s = s + sfpi::dst_reg[m(j, 2)];
            s = s + sfpi::dst_reg[m(j, 3)];
            sfpi::vFloat inv = recip(s);
#pragma GCC unroll 4
            for (uint32_t k = 0; k < n; ++k) {
                sfpi::dst_reg[m(j, k)] = sfpi::dst_reg[m(j, k)] * inv + bits(hc_eps);
            }
        }
        // Sinkhorn: column step, then iters - 1 (row, column) pairs
        col_normalize(hc_eps);
#pragma GCC unroll 0
        for (uint32_t it = 1; it < iters; ++it) {
            row_normalize(hc_eps);
            col_normalize(hc_eps);
        }
        sfpi::dst_reg++;  // the duplicate half (vectors 2k + 1)
    }
}

}  // namespace ckernel::sfpu
#endif

#ifdef MHC_POST_HC_COMPUTE
namespace mhc_post_hc {

// eltwise_chain element: runs the hc program in place over lane tiles Slot, Slot + 1.
template <compute_kernel_lib::Dst Slot>
struct HcProgram : compute_kernel_lib::UnaryOp<HcProgram<Slot>, Slot> {
    static constexpr uint32_t lane_width = compute_kernel_lib::to_u32(Slot) + LANE_TILES;
    const uint32_t* params;

    ALWI explicit HcProgram(const uint32_t* p) : params(p) {}

    static ALWI void init() {
        MATH((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::unused, DST_ACCUM_MODE>()));
    }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ::ckernel::sfpu::calculate_mhc_hc,
            compute_kernel_lib::to_u32(Slot) + slot_offset,
            VectorMode::RC_custom,
            params)));
    }
};

}  // namespace mhc_post_hc
#endif
