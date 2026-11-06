// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post collapse option (deepseek_v41_flash_d_p P.34, define MHC_POST_COLLAPSE): the next mHC step's work that
// mhc_pre_xing(deepseek=True, collapse_pre=) did, done inside the stats pass. The SFPU bodies are copies of
// ttnn/ttnn/bringup/mhc_pre_ttnn/kernels/mhc_pre_xing_compute.cpp (ymix, and coefficients + sinkhorn with
// MHC_XING_DEEPSEEK), line for line, so the results are bit-identical to that op; only where the scalars come from
// differs: common runtime args from RT0 (mhc_pre_xing: per-core args from 2), in mhc_pre_xing's scalar order: a_pre,
// a_post, a_res, inv_nc, norm_eps, hc_eps, lo, hi, iters, then the n (n + 2) biases. Keep the two in step.
//   ymix:          DEST 0 = the pre-block tile (slot 8 i + b, lane l = pre_i of token row 4 b + (l >> 3)), DEST 1..n
//                  = X'_i[c] -> DEST 1 = sum_i pre_i X'_i (one fp32 multiply, then fused multiply-adds in i order).
//   coefficients:  DEST 0 = a row [mix 0..n(n+2)-1 | sum x^2] coefficient-major (lane = token) -> [pre | post | logits]
//   sinkhorn:      logits -> comb (DeepSeek's order: softmax + eps, column norm, iters - 1 x {row, column}).

#pragma once

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace post_hc_sfpu {

using namespace sfpi;
using ckernel::sfpu::Converter;

template <int N, uint32_t RT0>
struct Hc {
    static constexpr uint32_t RT_A_PRE = RT0 + 0, RT_A_POST = RT0 + 1, RT_A_RES = RT0 + 2, RT_INV_NC = RT0 + 3,
                              RT_NORM_EPS = RT0 + 4, RT_HC_EPS = RT0 + 5, RT_ITERS = RT0 + 8, RT_BASE = RT0 + 9;
    static constexpr int MIX = N * (N + 2);  // coefficient slots; slot MIX holds sum x^2
    static constexpr int LOGIT0 = 2 * N;
    static constexpr int TILE_SLOTS = 32;
    static constexpr int m_slot(int i, int j) { return LOGIT0 + i * N + j; }
    static constexpr uint32_t pb_slot(uint32_t stream, uint32_t v) {
        return 8 * stream + 4 * (v >> 4) + ((v >> 1) & 3);
    }

    // 1/x for x > 0 finite: hardware seed + two Newton steps (mhc_pre_xing_compute.cpp recip_pos).
    static sfpi_inline vFloat recip_pos(vFloat x) {
        vFloat y = approx_recip(x);
        vFloat t = 2.0f - x * y;
        y = y * t;
        t = 2.0f - x * y;
        y = y * t;
        return y;
    }

    // 1/sqrt(x) for x > 0 finite: bit-trick seed + four Newton steps (mhc_pre_xing_compute.cpp rsqrt_pos).
    static sfpi_inline vFloat rsqrt_pos(vFloat x) {
        vInt i = as<vInt>(as<vUInt>(x) >> 1);
        vInt magic = 0x5f3759df;
        vFloat y = as<vFloat>(magic - i);
        vFloat half_x = x * 0.5f;
#pragma GCC unroll 0
        for (int it = 0; it < 4; ++it) {
            y = y * (1.5f - half_x * y * y);
        }
        return y;
    }

    // sigmoid(x) = 1 / (1 + exp(-x)); exp argument clamped so 1 + exp stays finite.
    static sfpi_inline vFloat sigmoid_acc(vFloat x) {
        vFloat z = -x;
        v_if(z > 80.0f) { z = 80.0f; }
        v_endif;
        vFloat e = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(z);
        return recip_pos(e + 1.0f);
    }

    // DEST tile 0: raw mixes + sum x^2 (coefficient-major) -> [pre | post | logits] (MHC_XING_DEEPSEEK).
    static void coefficients() {
        {
            vFloat inv_nc = Converter::as_float(get_common_arg_val<uint32_t>(RT_INV_NC));
            vFloat neps = Converter::as_float(get_common_arg_val<uint32_t>(RT_NORM_EPS));
            vFloat ssq = dst_reg[MIX];
            dst_reg[MIX] = rsqrt_pos(ssq * inv_nc + neps);
        }
#pragma GCC unroll 32
        for (int k = 0; k < MIX; ++k) {
            const uint32_t a_idx = k < N ? RT_A_PRE : (k < 2 * N ? RT_A_POST : RT_A_RES);
            vFloat r = dst_reg[MIX];
            vFloat a = Converter::as_float(get_common_arg_val<uint32_t>(a_idx));
            vFloat b = Converter::as_float(get_common_arg_val<uint32_t>(RT_BASE + k));
            vFloat z = dst_reg[k] * r;
            z = z * a + b;
            if (k < 2 * N) {
                vFloat s = sigmoid_acc(z);
                if (k >= N) {
                    s = s + s;
                } else {
                    s = s + Converter::as_float(get_common_arg_val<uint32_t>(RT_HC_EPS));
                }
                dst_reg[k] = s;
            } else {
                dst_reg[k] = z;
            }
        }
    }

    // DEST tile 0 slots [2n, 2n + n^2): logits -> comb (DeepSeek: softmax + eps, column norm, iters - 1 x {rows,
    // cols}).
    static void sinkhorn() {
        const vFloat eps = Converter::as_float(get_common_arg_val<uint32_t>(RT_HC_EPS));
        const uint32_t iters = get_common_arg_val<uint32_t>(RT_ITERS);
#pragma GCC unroll 4
        for (int i = 0; i < N; ++i) {
            vFloat mx = dst_reg[m_slot(i, 0)];
#pragma GCC unroll 4
            for (int j = 1; j < N; ++j) {
                mx = sfpi::max(mx, vFloat(dst_reg[m_slot(i, j)]));
            }
#pragma GCC unroll 4
            for (int j = 0; j < N; ++j) {
                vFloat d = dst_reg[m_slot(i, j)] - mx;
                dst_reg[m_slot(i, j)] = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(d);
            }
        }
#pragma GCC unroll 4
        for (int i = 0; i < N; ++i) {
            vFloat s = dst_reg[m_slot(i, 0)];
#pragma GCC unroll 4
            for (int j = 1; j < N; ++j) {
                s = s + dst_reg[m_slot(i, j)];
            }
            vFloat rs = recip_pos(s);
#pragma GCC unroll 4
            for (int j = 0; j < N; ++j) {
                dst_reg[m_slot(i, j)] = dst_reg[m_slot(i, j)] * rs + eps;
            }
        }
#pragma GCC unroll 4
        for (int j = 0; j < N; ++j) {
            vFloat s = dst_reg[m_slot(0, j)];
#pragma GCC unroll 4
            for (int i = 1; i < N; ++i) {
                s = s + dst_reg[m_slot(i, j)];
            }
            vFloat rc = recip_pos(s + eps);
#pragma GCC unroll 4
            for (int i = 0; i < N; ++i) {
                dst_reg[m_slot(i, j)] = dst_reg[m_slot(i, j)] * rc;
            }
        }
        const uint32_t loops = iters - 1;  // the host refuses iters < 1
#pragma GCC unroll 0
        for (uint32_t it = 0; it < loops; ++it) {
#pragma GCC unroll 4
            for (int i = 0; i < N; ++i) {
                vFloat s = dst_reg[m_slot(i, 0)];
#pragma GCC unroll 4
                for (int j = 1; j < N; ++j) {
                    s = s + dst_reg[m_slot(i, j)];
                }
                vFloat rs = recip_pos(s + eps);
#pragma GCC unroll 4
                for (int j = 0; j < N; ++j) {
                    dst_reg[m_slot(i, j)] = dst_reg[m_slot(i, j)] * rs;
                }
            }
#pragma GCC unroll 4
            for (int j = 0; j < N; ++j) {
                vFloat s = dst_reg[m_slot(0, j)];
#pragma GCC unroll 4
                for (int i = 1; i < N; ++i) {
                    s = s + dst_reg[m_slot(i, j)];
                }
                vFloat rc = recip_pos(s + eps);
#pragma GCC unroll 4
                for (int i = 0; i < N; ++i) {
                    dst_reg[m_slot(i, j)] = dst_reg[m_slot(i, j)] * rc;
                }
            }
        }
    }

    // DEST tile 0 = pre blocks, tiles 1..n = X'_i: tile 1 <- sum_i pre_i * X'_i.
    static void ymix() {
#pragma GCC unroll 8
        for (int v = 0; v < TILE_SLOTS; ++v) {
            vFloat acc = dst_reg[TILE_SLOTS + v] * dst_reg[pb_slot(0, v)];
#pragma GCC unroll 4
            for (int i = 1; i < N; ++i) {
                acc = dst_reg[TILE_SLOTS * (1 + i) + v] * dst_reg[pb_slot(i, v)] + acc;
            }
            dst_reg[TILE_SLOTS + v] = acc;
        }
    }
};

}  // namespace post_hc_sfpu
#endif
