// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_post stats compute (ttnn.bringup.mhc_post(..., stats_weight=W); host device/mhc_post_stats_program_factory.cpp).
//
// Per unit (block b of token tile-row r, mhc_post_stats_common.hpp), per column c (one DEST window, full sync):
//   DEST 0..CF-1  the unit's top-half coefficient tiles (cb_cf), DEST CF = F[c], DEST CF+1+j = X_j[c] (all
//                 UnpackToDestFp32, exact);
//   SFPU          per data vector v of the top half (rows 0..15): every input loaded once, acc_k = F post_k (one
//                 rounding), then acc_k = X_j w_jk + acc_k (fused multiply-adds, j in order) for every output k: the
//                 arithmetic and term order of mhc_post's WeightedSum, so X' is bit-identical to mhc_post; then
//                 ss[v] = sum_k acc_k^2; acc_k -> DEST CF+k, ss -> DEST CF+n;
//   then the bottom-half coefficient tiles over DEST 0..CF-1 (an unpack-to-dest copy waits for the SFPU) and the
//   bottom half (vectors 16..31); pack X'_k -> cb_out (the writer) and cb_xs (this kernel), ss -> cb_ssc.
// After the unit's columns, one DEST window: DEST 0 = sum over (c, k) X'_k[c] @ W[k, c] (matmul_tiles: X' = in0 /
//   srcB, W = in1 / srcA, as ttnn.matmul(x, fn_t); fp32 DEST, the program's fidelity), DEST 1 = sum_c ss[c] (fp32
//   SFPU adds in column order) -> cb_mfin, cb_sfin. The writer re-lays the ss tile coefficient-major (cb_soa; lane =
//   token row), this kernel sums its 32 column slots in order (cb_soa_out), the writer puts the totals in column
//   n (n + 2) of the partial and sends it to the row's root.
// Root phase: per rooted row, its NB partials (cb_part, block order) summed in fp32 on the SFPU -> cb_fin.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "mhc_post_stats_common.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

constexpr uint32_t n_streams = get_compile_time_arg_val(0);
constexpr uint32_t ct = get_compile_time_arg_val(1);
constexpr uint32_t rt = get_compile_time_arg_val(2);
constexpr uint32_t bw = get_compile_time_arg_val(3);
constexpr uint32_t nb = get_compile_time_arg_val(4);
constexpr uint32_t cf = get_compile_time_arg_val(6);
constexpr uint32_t cb_in = get_compile_time_arg_val(7);
constexpr uint32_t cb_w = get_compile_time_arg_val(8);
constexpr uint32_t cb_cf = get_compile_time_arg_val(9);
constexpr uint32_t cb_out = get_compile_time_arg_val(10);
constexpr uint32_t cb_xs = get_compile_time_arg_val(11);
constexpr uint32_t cb_ssc = get_compile_time_arg_val(12);
constexpr uint32_t cb_mfin = get_compile_time_arg_val(13);
constexpr uint32_t cb_sfin = get_compile_time_arg_val(14);
constexpr uint32_t cb_soa = get_compile_time_arg_val(15);
constexpr uint32_t cb_soa_out = get_compile_time_arg_val(16);
constexpr uint32_t cb_part = get_compile_time_arg_val(17);
constexpr uint32_t cb_fin = get_compile_time_arg_val(18);

#ifdef TRISC_MATH
namespace stats_sfpu {

using namespace sfpi;

constexpr int N = static_cast<int>(n_streams);
constexpr int CF = static_cast<int>(cf);
constexpr int T = 32;  // vectors per tile

template <int V>
sfpi_inline void mix_vector() {
    constexpr int q4 = (V >> 1) & 3;  // row block within the half
    vFloat acc[N];
    {
        const vFloat d = dst_reg[T * CF + V];  // F
#pragma GCC unroll 4
        for (int k = 0; k < N; ++k) {
            acc[k] = d * vFloat(dst_reg[4 * (k * (N + 1)) + q4]);
        }
    }
#pragma GCC unroll 4
    for (int j = 0; j < N; ++j) {
        const vFloat d = dst_reg[T * (CF + 1 + j) + V];  // X_j
#pragma GCC unroll 4
        for (int k = 0; k < N; ++k) {
            acc[k] = d * vFloat(dst_reg[4 * (k * (N + 1) + 1 + j) + q4]) + acc[k];
        }
    }
    vFloat s = acc[0] * acc[0];
#pragma GCC unroll 4
    for (int k = 1; k < N; ++k) {
        s = acc[k] * acc[k] + s;
    }
#pragma GCC unroll 4
    for (int k = 0; k < N; ++k) {
        dst_reg[T * (CF + k) + V] = acc[k];
    }
    dst_reg[T * (CF + N) + V] = s;
}

template <int V0, int... Vs>
sfpi_inline void mix_vectors(std::integer_sequence<int, Vs...>) {
    (mix_vector<V0 + Vs>(), ...);
}

template <int H>
void mix_half() {
    mix_vectors<16 * H>(std::make_integer_sequence<int, 16>{});
}

// DEST tile A += DEST tile B, lane-wise (fp32).
template <int A, int B>
void add_tiles() {
#pragma GCC unroll 8
    for (int v = 0; v < T; ++v) {
        dst_reg[T * A + v] = vFloat(dst_reg[T * A + v]) + vFloat(dst_reg[T * B + v]);
    }
}

// DEST 0 slot 0 <- sum of slots 0..31 (column order), lane-wise.
void slot_total() {
    vFloat s = dst_reg[0];
#pragma GCC unroll 31
    for (int k = 1; k < T; ++k) {
        s = s + dst_reg[k];
    }
    dst_reg[0] = s;
}

}  // namespace stats_sfpu
#endif

using namespace ckernel;

ALWI void sfpu_init() { MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

// One custom SFPU pass over DEST from tile 0 (the function addresses DEST tiles absolutely).
#define STATS_SFPU(fn)                                                         \
    do {                                                                       \
        sfpu_init();                                                           \
        MATH((_llk_math_eltwise_unary_sfpu_params_(fn, 0, VectorMode::None))); \
    } while (0)

ALWI void copy_from(uint32_t cb, uint32_t tile, uint32_t dst) { copy_tile(cb, tile, dst); }

void kernel_main() {
    constexpr uint32_t n = n_streams;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t n_rooted = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_in, cb_out);

    uint32_t cur_b = 0xFFFFFFFFu;
    for (uint32_t u = start; u < start + count; ++u) {
        const mhc_stats::Unit U = mhc_stats::unit_of(u, rt, bw, ct);
        if (U.b != cur_b) {
            if (cur_b != 0xFFFFFFFFu) {
                cb_pop_front(cb_w, n * bw);
            }
            cb_wait_front(cb_w, n * bw);
            cur_b = U.b;
        }
        cb_wait_front(cb_cf, 2 * cf);
        cb_wait_front(cb_in, (n + 1) * bw);
        cb_reserve_back(cb_out, n * bw);
        cb_reserve_back(cb_xs, n * bw);
        cb_reserve_back(cb_ssc, bw);
        for (uint32_t c = 0; c < U.valid; ++c) {
            tile_regs_acquire();
            copy_init(cb_cf);  // srcA is on the fp32 unpack-to-dest format (hw_startup / after the matmul below)
            for (uint32_t t = 0; t < cf; ++t) {
                copy_from(cb_cf, t, t);
            }
            copy_init(cb_in);
            for (uint32_t t = 0; t <= n; ++t) {
                copy_from(cb_in, c * (n + 1) + t, cf + t);
            }
            STATS_SFPU(stats_sfpu::mix_half<0>);
            copy_init(cb_cf);
            for (uint32_t t = 0; t < cf; ++t) {
                copy_from(cb_cf, cf + t, t);
            }
            STATS_SFPU(stats_sfpu::mix_half<1>);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < n; ++k) {
                pack_tile<true>(cf + k, cb_out, c * n + k);
            }
            for (uint32_t k = 0; k < n; ++k) {
                pack_tile<true>(cf + k, cb_xs, c * n + k);
            }
            pack_tile<true>(cf + n, cb_ssc, c);
            tile_regs_release();
        }
        // whole batches (B columns) so every CB index stays inside its window
        cb_push_back(cb_out, n * bw);
        cb_push_back(cb_xs, n * bw);
        cb_push_back(cb_ssc, bw);
        cb_pop_front(cb_in, (n + 1) * bw);
        cb_pop_front(cb_cf, 2 * cf);

        // the unit's partial: DEST 0 = X'_unit @ W_block (FPU), DEST 1 = sum of the columns' squares (SFPU)
        cb_wait_front(cb_xs, n * bw);
        cb_wait_front(cb_ssc, bw);
        cb_reserve_back(cb_mfin, 1);
        cb_reserve_back(cb_sfin, 1);
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_xs, cb_w);
        matmul_init(cb_xs, cb_w);
        for (uint32_t c = 0; c < U.valid; ++c) {
            for (uint32_t k = 0; k < n; ++k) {
                matmul_tiles(cb_xs, cb_w, c * n + k, k * bw + c, 0);
            }
        }
        reconfig_data_format_srca(cb_ssc);
        copy_init(cb_ssc);
        copy_from(cb_ssc, 0, 1);
        for (uint32_t c = 1; c < U.valid; ++c) {
            copy_from(cb_ssc, c, 2);
            STATS_SFPU((stats_sfpu::add_tiles<1, 2>));
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_mfin);
        pack_tile(1, cb_sfin);
        tile_regs_release();
        cb_push_back(cb_mfin, 1);
        cb_push_back(cb_sfin, 1);
        cb_pop_front(cb_xs, n * bw);
        cb_pop_front(cb_ssc, bw);

        // row totals of the unit's sum-of-squares tile (coefficient-major from the writer)
        cb_wait_front(cb_soa, 1);
        cb_reserve_back(cb_soa_out, 1);
        tile_regs_acquire();
        copy_init(cb_soa);
        copy_from(cb_soa, 0, 0);
        STATS_SFPU(stats_sfpu::slot_total);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_soa_out);
        tile_regs_release();
        cb_push_back(cb_soa_out, 1);
        cb_pop_front(cb_soa, 1);
    }
    if (cur_b != 0xFFFFFFFFu) {
        cb_pop_front(cb_w, n * bw);
    }

    // root phase: each rooted row's partials summed in block order
    for (uint32_t i = 0; i < n_rooted; ++i) {
        cb_reserve_back(cb_fin, 1);
        tile_regs_acquire();
        copy_init(cb_part);
        cb_wait_front(cb_part, nb);
        copy_from(cb_part, 0, 0);
        for (uint32_t b = 1; b < nb; ++b) {
            copy_from(cb_part, b, 1);
            STATS_SFPU((stats_sfpu::add_tiles<0, 1>));
        }
        cb_pop_front(cb_part, nb);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_fin);
        tile_regs_release();
        cb_push_back(cb_fin, 1);
    }
}
