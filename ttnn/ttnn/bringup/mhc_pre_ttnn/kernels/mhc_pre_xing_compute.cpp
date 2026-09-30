// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing compute: the Xing4.0 mHC coefficients from an all-reduced mix row, and the collapse.
//
// Coefficient stage (compute_coef), one coefficient-major tile per token tile-row (the writer's scatter of the
// row [mix 0..n(n+2)-1 | sum x^2]; lane = token):
//   r    = rsqrt(sum x^2 * inv_nc + norm_eps)
//   pre  = sigmoid(a_pre * mix * r + b)                    (no + eps)
//   post = 2 sigmoid(a_post * mix * r + b)
//   L    = clamp(a_res * mix * r + b, lo, hi)             (L[i][j] at slot 2n + i n + j)
//   comb = exp(L - rowmax_j L); iters x { rows: m / (rowsum + eps); columns: m / (colsum + eps) }
// All lane-wise fp32 SFPU math: accurate exp, Newton-refined reciprocal / rsqrt, sums in index order.
//
// y-mix (has_streams): per y column tile, DEST 0 = the pre-block tile, DEST 1..n = X_i[c] (UnpackToDestFp32, exact);
// y = sum_i pre_i * X_i on the SFPU (one fp32 multiply, then fused multiply-adds in stream order) -> DEST 1 -> cb_y.
// dst_full_sync_en (8 fp32 DEST tiles) holds the n + 1 tiles.
//
// pack_stats: per token tile-row, DEST 0 += X_k * X_k over every stream tile k of the row (chunks of `chunk` tiles in
// DEST 1..chunk, one DEST window for the whole row; exact fp32 unpack, fused multiply-adds in k order) -> cb_acc; the
// writer's coefficient-major copy of it (cb_soa, lane = token row) -> slot 0 = sum of its 32 slots in column order ->
// cb_soa_out.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "mhc_pre_xing_common.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

constexpr uint32_t n_streams = get_compile_time_arg_val(0);
constexpr uint32_t ct = get_compile_time_arg_val(1);
constexpr bool compute_coef = get_compile_time_arg_val(2) != 0;
constexpr bool has_streams = get_compile_time_arg_val(3) != 0;
constexpr uint32_t cb_soa = get_compile_time_arg_val(4);
constexpr uint32_t cb_soa_out = get_compile_time_arg_val(5);
constexpr uint32_t cb_pb = get_compile_time_arg_val(6);
constexpr uint32_t cb_x = get_compile_time_arg_val(7);
constexpr uint32_t cb_y = get_compile_time_arg_val(8);
constexpr bool pack_stats = get_compile_time_arg_val(9) != 0;
constexpr uint32_t chunk = get_compile_time_arg_val(10);
constexpr uint32_t cb_acc = get_compile_time_arg_val(11);

// Runtime args: 0 start, 1 count, then the scalars (fp32 bits), then the n (n + 2) biases.
constexpr uint32_t RT_A_PRE = 2, RT_A_POST = 3, RT_A_RES = 4, RT_INV_NC = 5, RT_NORM_EPS = 6, RT_HC_EPS = 7, RT_LO = 8,
                   RT_HI = 9, RT_ITERS = 10, RT_BASE = 11;

#ifdef TRISC_MATH
namespace xing_sfpu {

using namespace sfpi;
using ckernel::sfpu::Converter;

constexpr int N = static_cast<int>(n_streams);
constexpr int MIX = N * (N + 2);  // coefficient slots; slot MIX holds sum x^2
constexpr int LOGIT0 = 2 * N;
constexpr int TILE_SLOTS = 32;
constexpr int m_slot(int i, int j) { return LOGIT0 + i * N + j; }

// 1/x for x > 0 finite: hardware seed + two Newton steps (~fp32 accurate) (mhc_pre_compute.cpp recip_pos).
sfpi_inline vFloat recip_pos(vFloat x) {
    vFloat y = approx_recip(x);
    vFloat t = 2.0f - x * y;
    y = y * t;
    t = 2.0f - x * y;
    y = y * t;
    return y;
}

// 1/sqrt(x) for x > 0 finite: bit-trick seed + four Newton steps (mhc_pre_compute.cpp rsqrt_pos).
sfpi_inline vFloat rsqrt_pos(vFloat x) {
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
sfpi_inline vFloat sigmoid_acc(vFloat x) {
    vFloat z = -x;
    v_if(z > 80.0f) { z = 80.0f; }
    v_endif;
    vFloat e = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(z);
    return recip_pos(e + 1.0f);
}

// DEST tile 0: raw mixes + sum x^2 (coefficient-major) -> [pre | post | logits].
void coefficients() {
    {
        vFloat inv_nc = Converter::as_float(get_arg_val<uint32_t>(RT_INV_NC));
        vFloat neps = Converter::as_float(get_arg_val<uint32_t>(RT_NORM_EPS));
        vFloat ssq = dst_reg[MIX];
        dst_reg[MIX] = rsqrt_pos(ssq * inv_nc + neps);
    }
#pragma GCC unroll 32
    for (int k = 0; k < MIX; ++k) {
        const uint32_t a_idx = k < N ? RT_A_PRE : (k < 2 * N ? RT_A_POST : RT_A_RES);
        vFloat r = dst_reg[MIX];
        vFloat a = Converter::as_float(get_arg_val<uint32_t>(a_idx));
        vFloat b = Converter::as_float(get_arg_val<uint32_t>(RT_BASE + k));
        vFloat z = dst_reg[k] * r;
        z = z * a + b;
        if (k < 2 * N) {
            vFloat s = sigmoid_acc(z);
            if (k >= N) {
                s = s + s;
            }
            dst_reg[k] = s;
        } else {
            vFloat lo = Converter::as_float(get_arg_val<uint32_t>(RT_LO));
            vFloat hi = Converter::as_float(get_arg_val<uint32_t>(RT_HI));
            v_if(z < lo) { z = lo; }
            v_endif;
            v_if(z > hi) { z = hi; }
            v_endif;
            dst_reg[k] = z;
        }
    }
}

// DEST tile 0 slots [2n, 2n + n^2): logits -> comb (Xing order: exp(L - rowmax), iters x {row, column}).
void sinkhorn() {
    const vFloat eps = Converter::as_float(get_arg_val<uint32_t>(RT_HC_EPS));
    const uint32_t iters = get_arg_val<uint32_t>(RT_ITERS);
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
#pragma GCC unroll 0
    for (uint32_t it = 0; it < iters; ++it) {
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

// DEST tile 0 = pre blocks, tiles 1..n = X_i: tile 1 <- sum_i pre_i * X_i.
void ymix() {
#pragma GCC unroll 8
    for (int v = 0; v < TILE_SLOTS; ++v) {
        vFloat acc = dst_reg[TILE_SLOTS + v] * dst_reg[mhc_xing::pb_slot(0, v)];
#pragma GCC unroll 4
        for (int i = 1; i < N; ++i) {
            acc = dst_reg[TILE_SLOTS * (1 + i) + v] * dst_reg[mhc_xing::pb_slot(i, v)] + acc;
        }
        dst_reg[TILE_SLOTS + v] = acc;
    }
}

// DEST 0 (+)= sum_{t < chunk} tile(1 + t)^2, lane-wise.
template <bool FIRST>
void sumsq_chunk() {
#pragma GCC unroll 4
    for (int v = 0; v < TILE_SLOTS; ++v) {
        vFloat acc;
        int t0 = 0;
        if constexpr (FIRST) {
            vFloat x = dst_reg[TILE_SLOTS + v];
            acc = x * x;
            t0 = 1;
        } else {
            acc = dst_reg[v];
        }
#pragma GCC unroll 7
        for (int t = t0; t < static_cast<int>(chunk); ++t) {
            vFloat x = dst_reg[TILE_SLOTS * (1 + t) + v];
            acc = x * x + acc;
        }
        dst_reg[v] = acc;
    }
}

// DEST 0 slot 0 <- sum of slots 0..31 (column order), lane-wise.
void slot_total() {
    vFloat s = dst_reg[0];
#pragma GCC unroll 31
    for (int k = 1; k < TILE_SLOTS; ++k) {
        s = s + dst_reg[k];
    }
    dst_reg[0] = s;
}

}  // namespace xing_sfpu
#endif

using namespace ckernel;

ALWI void custom_sfpu_init() { MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    if constexpr (pack_stats) {
        constexpr uint32_t k_tiles = n_streams * ct;
        compute_kernel_hw_startup(cb_x, cb_acc);
        for (uint32_t r = 0; r < count; ++r) {
            cb_reserve_back(cb_acc, 1);
            pack_reconfig_data_format(cb_acc);
            tile_regs_acquire();
            for (uint32_t k0 = 0; k0 < k_tiles; k0 += chunk) {
                cb_wait_front(cb_x, chunk);
                copy_tile_to_dst_init_short(cb_x);
                for (uint32_t t = 0; t < chunk; ++t) {
                    copy_tile(cb_x, t, 1 + t);
                }
                custom_sfpu_init();
                if (k0 == 0) {
                    MATH((_llk_math_eltwise_unary_sfpu_params_(xing_sfpu::sumsq_chunk<true>, 0, VectorMode::None)));
                } else {
                    MATH((_llk_math_eltwise_unary_sfpu_params_(xing_sfpu::sumsq_chunk<false>, 0, VectorMode::None)));
                }
                cb_pop_front(cb_x, chunk);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_acc);
            tile_regs_release();
            cb_push_back(cb_acc, 1);

            cb_wait_front(cb_soa, 1);
            cb_reserve_back(cb_soa_out, 1);
            pack_reconfig_data_format(cb_soa_out);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_soa);
            copy_tile(cb_soa, 0, 0);
            custom_sfpu_init();
            MATH((_llk_math_eltwise_unary_sfpu_params_(xing_sfpu::slot_total, 0, VectorMode::None)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_soa_out);
            tile_regs_release();
            cb_push_back(cb_soa_out, 1);
            cb_pop_front(cb_soa, 1);
        }
        return;
    }

    constexpr uint32_t cb_first_in = compute_coef ? cb_soa : cb_x;
    constexpr uint32_t cb_first_out = compute_coef ? cb_soa_out : cb_y;
    compute_kernel_hw_startup(cb_first_in, cb_first_out);

    mhc_xing::SegmentWalker walker(start, count, has_streams ? ct : 1);
    while (!walker.done()) {
        const mhc_xing::Segment seg = walker.next();
        if constexpr (compute_coef) {
            cb_wait_front(cb_soa, 1);
            cb_reserve_back(cb_soa_out, 1);
            pack_reconfig_data_format(cb_soa_out);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_soa);
            copy_tile(cb_soa, 0, 0);
            custom_sfpu_init();
            MATH((_llk_math_eltwise_unary_sfpu_params_(xing_sfpu::coefficients, 0, VectorMode::None)));
            MATH((_llk_math_eltwise_unary_sfpu_params_(xing_sfpu::sinkhorn, 0, VectorMode::None)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_soa_out);
            tile_regs_release();
            cb_push_back(cb_soa_out, 1);
            cb_pop_front(cb_soa, 1);
        }
        if constexpr (has_streams) {
            cb_wait_front(cb_pb, 1);
            pack_reconfig_data_format(cb_y);
            for (uint32_t c = 0; c < seg.cols; ++c) {
                cb_wait_front(cb_x, n_streams);
                cb_reserve_back(cb_y, 1);
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_pb);
                copy_tile(cb_pb, 0, 0);
                copy_tile_to_dst_init_short(cb_x);
                for (uint32_t i = 0; i < n_streams; ++i) {
                    copy_tile(cb_x, i, 1 + i);
                }
                custom_sfpu_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(xing_sfpu::ymix, 0, VectorMode::None)));
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(1, cb_y);
                tile_regs_release();
                cb_push_back(cb_y, 1);
                cb_pop_front(cb_x, n_streams);
            }
            cb_pop_front(cb_pb, 1);
        }
    }
}
