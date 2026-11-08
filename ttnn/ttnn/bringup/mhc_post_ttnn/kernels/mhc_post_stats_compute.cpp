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
// MHC_POST_COLLAPSE (P.34, bf16 X): after a unit's columns, per column c one DEST window: DEST 0 = the unit's pre-block
//   tile (cb_pb, UnpackToDestFp32), DEST 1..n = the bf16 X'_k[c] just packed to cb_xs (exact in the fp32 DEST), ymix
//   -> DEST 1 -> cb_y (bf16: the packer's one rounding) = mhc_pre_xing(collapse_pre=)'s y of the bf16 X'. Root phase:
//   after each row's sum, its coefficient-major copy (cb_hsoa, from the writer) -> coefficients + Sinkhorn (DeepSeek)
//   -> cb_hsoa_out = mhc_pre_xing(deepseek=True)'s hc of that row (kernels/mhc_post_hc_sfpu.hpp).
// MHC_POST_NORM (P.35, with the collapse): the unit's bf16 y tiles stay in cb_y (sized for every unit of the core);
//   after them, one DEST window sums their squares lane-wise (exact bf16 values in the fp32 DEST, column order) and
//   folds each row's 32 columns to 8 (lane-wise adds of the 4 vectors sharing its lanes) -> cb_nsq; the writer turns
//   the 8 columns coefficient-major, slot_total8 sums them (cb_soa_out, taken before the X' squares'), and the writer
//   puts the totals in the partial's column n (n + 2) + 1. Root phase: in the window of a row's partial sum, the
//   rsqrt(sum / C + eps) of that column (the hc's Newton rsqrt, DEST 1, row-major) -> cb_nrs (the reader sends the 32
//   values to the row's units). Then the rooted rows' hc (as without the norm), then the norm phase, per unit: DEST 0 =
//   its row's rstd tile (cb_nr), DEST 1 = its weight tile (cb_ng), DEST 2..7 = up to 6 y columns -> (y rstd) w fp32 on
//   the SFPU -> cb_yn (bf16: one rounding).

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
#ifdef MHC_POST_COLLAPSE
#include "mhc_post_hc_sfpu.hpp"
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
#ifdef MHC_POST_STATS_F16
constexpr uint32_t cb_f = get_compile_time_arg_val(19);  // P.19: bf16 F tiles (UnpackToDestFp32: exact in DEST)
#endif
#ifdef MHC_POST_COLLAPSE
constexpr uint32_t cb_pb = get_compile_time_arg_val(20);  // P.34 (bf16 X, so bf16 F: cb_f is at 19)
constexpr uint32_t cb_y = get_compile_time_arg_val(21);
constexpr uint32_t cb_hsoa = get_compile_time_arg_val(22);
constexpr uint32_t cb_hsoa_out = get_compile_time_arg_val(23);
#endif
#ifdef MHC_POST_NORM
constexpr uint32_t cb_nsq = get_compile_time_arg_val(24);
constexpr uint32_t cb_nr = get_compile_time_arg_val(25);
constexpr uint32_t cb_ng = get_compile_time_arg_val(26);
constexpr uint32_t cb_yn = get_compile_time_arg_val(27);
constexpr uint32_t cb_nrs = get_compile_time_arg_val(28);
#endif

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

#ifdef MHC_POST_NORM
// DEST tile 0 <- tile 0 squared (lane-wise)
void sq_first() {
#pragma GCC unroll 8
    for (int v = 0; v < T; ++v) {
        const vFloat a = dst_reg[v];
        dst_reg[v] = a * a;
    }
}

// DEST tile 0 <- tile 1 squared + tile 0 (lane-wise)
void sq_acc() {
#pragma GCC unroll 8
    for (int v = 0; v < T; ++v) {
        const vFloat b = dst_reg[T + v];
        dst_reg[v] = b * b + vFloat(dst_reg[v]);
    }
}

// DEST tile 0: the 4 vectors of each (row block, lane) - columns c, c + 1, c + 16, c + 17, c = 2 (l & 7) - summed into
// the first (vector k with k & 9 == 0): row i's 32 columns -> 8 values, at row-major columns 0, 2, .., 14
template <int K>
sfpi_inline void fold_vector() {
    dst_reg[K] = (vFloat(dst_reg[K]) + vFloat(dst_reg[K + 1])) + (vFloat(dst_reg[K + 8]) + vFloat(dst_reg[K + 9]));
}
void sq_fold() {
    fold_vector<0>();
    fold_vector<2>();
    fold_vector<4>();
    fold_vector<6>();
    fold_vector<16>();
    fold_vector<18>();
    fold_vector<20>();
    fold_vector<22>();
}

// DEST 0 slot 0 <- sum of slots 0..7 (lane-wise; the folded y^2 tile, coefficient-major)
void slot_total8() {
    vFloat s = dst_reg[0];
#pragma GCC unroll 7
    for (int k = 1; k < 8; ++k) {
        s = s + dst_reg[k];
    }
    dst_reg[0] = s;
}

// DEST tile 2 + C % 6 <- (it * rstd) * w: rstd from tile 0 slot blk(v), w from tile 1 slot 4 C + g(v) (every DEST index
// a constant: one instance per vector, DS41-189)
template <int C, int V>
sfpi_inline void norm_vector() {
    constexpr int D = 2 + C % 6;  // y's DEST tile: columns 0..5 in the first window, 6..7 in the second
    constexpr int blk = 4 * (V >> 4) + ((V >> 1) & 3);
    constexpr int g = (V & 1) | (((V >> 3) & 1) << 1);
    const vFloat y = dst_reg[D * T + V];
    const vFloat r = dst_reg[blk];
    const vFloat w = dst_reg[T + 4 * C + g];
    dst_reg[D * T + V] = (y * r) * w;
}

template <int C, int... Vs>
sfpi_inline void norm_vectors(std::integer_sequence<int, Vs...>) {
    (norm_vector<C, Vs>(), ...);
}

template <int C>
void norm_apply() {
    norm_vectors<C>(std::make_integer_sequence<int, T>{});
}
#endif

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

#if defined(MHC_POST_COLLAPSE) && defined(TRISC_MATH)
using HcK = post_hc_sfpu::Hc<static_cast<int>(n_streams), 0>;  // the scalars: common runtime args 0..
#endif
#if defined(MHC_POST_NORM) && defined(TRISC_MATH)
// P.35: DEST 0 = a root's summed partial row (row-major) -> DEST 1 = rsqrt(sum * inv_c + eps) on the vectors holding
// column MIX + 1 (the sum of y^2; k & 1 == 1, k & 8 == 8: columns 17 + 2 (l & 7)); scalars after the hc's (common args)
template <int K>
sfpi_inline void rstd_vector(sfpi::vFloat inv_c, sfpi::vFloat eps) {
    using namespace sfpi;
    dst_reg[32 + K] = HcK::rsqrt_pos(vFloat(dst_reg[K]) * inv_c + eps);
}
void norm_rstd() {
    using namespace sfpi;
    static_assert(n_streams * (n_streams + 2) + 1 >= 17, "the sum of y^2 sits in columns 17..31");
    constexpr uint32_t RT_NORM = 9 + n_streams * (n_streams + 2);
    const vFloat inv_c = ckernel::sfpu::Converter::as_float(get_common_arg_val<uint32_t>(RT_NORM));
    const vFloat eps = ckernel::sfpu::Converter::as_float(get_common_arg_val<uint32_t>(RT_NORM + 1));
    rstd_vector<9>(inv_c, eps);
    rstd_vector<11>(inv_c, eps);
    rstd_vector<13>(inv_c, eps);
    rstd_vector<15>(inv_c, eps);
    rstd_vector<25>(inv_c, eps);
    rstd_vector<27>(inv_c, eps);
    rstd_vector<29>(inv_c, eps);
    rstd_vector<31>(inv_c, eps);
}

#endif
#ifdef MHC_POST_NORM
template <int C>
ALWI void norm_apply_col() {
    STATS_SFPU(stats_sfpu::norm_apply<C>);
}
#endif

// MHC_POST_STATS_X16 (P.20: bf16 X in cb_in, bf16 X' in cb_out / cb_xs): srcA / the packer to ``cb``'s format where
// the CBs alternate fp32 / bf16; absent otherwise (the fp32 program does no reconfiguration).
ALWI void srca_to(uint32_t cb) {
#ifdef MHC_POST_STATS_X16
    reconfig_data_format_srca(cb);
#endif
}
ALWI void pack_to(uint32_t cb) {
#ifdef MHC_POST_STATS_X16
    pack_reconfig_data_format(cb);
#endif
}

#ifdef MHC_POST_COLLAPSE
// The next step's hc of a rooted row (mhc_pre_xing(deepseek=True)'s coefficient stage on the summed row): its
// coefficient-major copy (cb_hsoa front) -> cb_hsoa_out.
ALWI void hc_row() {
    cb_wait_front(cb_hsoa, 1);
    cb_reserve_back(cb_hsoa_out, 1);
    tile_regs_acquire();
    copy_init(cb_hsoa);
    copy_from(cb_hsoa, 0, 0);
    sfpu_init();
    MATH((_llk_math_eltwise_unary_sfpu_params_(HcK::coefficients, 0, VectorMode::None)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(HcK::sinkhorn, 0, VectorMode::None)));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_hsoa_out);
    tile_regs_release();
    cb_push_back(cb_hsoa_out, 1);
    cb_pop_front(cb_hsoa, 1);
}
#endif

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
#ifdef MHC_POST_STATS_F16
        cb_wait_front(cb_f, bw);
#endif
        cb_reserve_back(cb_out, n * bw);
        cb_reserve_back(cb_xs, n * bw);
        cb_reserve_back(cb_ssc, bw);
        for (uint32_t c = 0; c < U.valid; ++c) {
            tile_regs_acquire();
            srca_to(cb_cf);
            copy_init(cb_cf);  // srcA is on the fp32 unpack-to-dest format (hw_startup / after the matmul below)
            for (uint32_t t = 0; t < cf; ++t) {
                copy_from(cb_cf, t, t);
            }
#ifdef MHC_POST_STATS_F16
            reconfig_data_format_srca(cb_cf, cb_f);
            copy_init(cb_f);
            copy_from(cb_f, c, cf);  // F (bf16 -> fp32 DEST, exact)
            reconfig_data_format_srca(cb_f, cb_in);
            copy_init(cb_in);
            for (uint32_t t = 1; t <= n; ++t) {
                copy_from(cb_in, c * (n + 1) + t, cf + t);
            }
#else
            copy_init(cb_in);
            for (uint32_t t = 0; t <= n; ++t) {
                copy_from(cb_in, c * (n + 1) + t, cf + t);
            }
#endif
            STATS_SFPU(stats_sfpu::mix_half<0>);
            srca_to(cb_cf);
            copy_init(cb_cf);
            for (uint32_t t = 0; t < cf; ++t) {
                copy_from(cb_cf, cf + t, t);
            }
            STATS_SFPU(stats_sfpu::mix_half<1>);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(cb_out);
            for (uint32_t k = 0; k < n; ++k) {
                pack_tile<true>(cf + k, cb_out, c * n + k);
            }
            for (uint32_t k = 0; k < n; ++k) {
                pack_tile<true>(cf + k, cb_xs, c * n + k);
            }
            pack_to(cb_ssc);
            pack_tile<true>(cf + n, cb_ssc, c);
            tile_regs_release();
        }
        // whole batches (B columns) so every CB index stays inside its window
        cb_push_back(cb_out, n * bw);
        cb_push_back(cb_xs, n * bw);
        cb_push_back(cb_ssc, bw);
        cb_pop_front(cb_in, (n + 1) * bw);
#ifdef MHC_POST_STATS_F16
        cb_pop_front(cb_f, bw);
#endif
        cb_pop_front(cb_cf, 2 * cf);

#ifdef MHC_POST_COLLAPSE
        // the next step's collapse of the unit's bf16 X' columns: y = sum_k pre_k X'_k (mhc_pre_xing's ymix)
        cb_wait_front(cb_xs, n * bw);
        cb_wait_front(cb_pb, 1);
        cb_reserve_back(cb_y, bw);
        pack_reconfig_data_format(cb_y);
        for (uint32_t c = 0; c < U.valid; ++c) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_pb);
            copy_init(cb_pb);
            copy_from(cb_pb, 0, 0);
            reconfig_data_format_srca(cb_pb, cb_xs);
            copy_init(cb_xs);
            for (uint32_t k = 0; k < n; ++k) {
                copy_from(cb_xs, c * n + k, 1 + k);
            }
            STATS_SFPU(HcK::ymix);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(1, cb_y, c);
            tile_regs_release();
        }
        cb_push_back(cb_y, bw);
        cb_pop_front(cb_pb, 1);
#ifdef MHC_POST_NORM
        {  // the unit's lane-wise sum of y^2 (the y tiles stay in cb_y: unit j's at j B from the front)
            const uint32_t j = u - start;
            cb_wait_front(cb_y, (j + 1) * bw);
            cb_reserve_back(cb_nsq, 1);
            pack_reconfig_data_format(cb_nsq);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_y);
            copy_init(cb_y);
            copy_from(cb_y, j * bw, 0);
            STATS_SFPU(stats_sfpu::sq_first);
            for (uint32_t c = 1; c < U.valid; ++c) {
                copy_from(cb_y, j * bw + c, 1);
                STATS_SFPU(stats_sfpu::sq_acc);
            }
            STATS_SFPU(stats_sfpu::sq_fold);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_nsq);
            tile_regs_release();
            cb_push_back(cb_nsq, 1);
        }
#endif
        pack_reconfig_data_format(cb_mfin);
#endif

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

#ifdef MHC_POST_NORM
        // first the y^2 tile's row totals (the writer turns its 8 folded columns around while the matmul runs)
        cb_wait_front(cb_soa, 1);
        cb_reserve_back(cb_soa_out, 1);
        tile_regs_acquire();
        copy_init(cb_soa);
        copy_from(cb_soa, 0, 0);
        STATS_SFPU(stats_sfpu::slot_total8);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_soa_out);
        tile_regs_release();
        cb_push_back(cb_soa_out, 1);
        cb_pop_front(cb_soa, 1);
#endif
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
#ifdef MHC_POST_NORM
    if (n_rooted) {  // (a root with no units of its own still has hw_startup's bf16 formats)
        reconfig_data_format_srca(cb_part);
        pack_reconfig_data_format(cb_fin);
    }
#endif
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
#ifdef MHC_POST_NORM
        // the row's rstd from the sum just made (DEST 1, column n (n + 2) + 1) -> cb_nrs: the writer sends it before
        // any transpose (the hc's coefficient-major copy comes after)
        cb_reserve_back(cb_nrs, 1);
        STATS_SFPU(norm_rstd);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_fin);
#ifdef MHC_POST_NORM
        pack_tile(1, cb_nrs);
#endif
        tile_regs_release();
        cb_push_back(cb_fin, 1);
#ifdef MHC_POST_NORM
        cb_push_back(cb_nrs, 1);
#endif
#if defined(MHC_POST_NORM)
        // (the rstd went out with the sum: cb_nrs, above)
#elif defined(MHC_POST_COLLAPSE)
        hc_row();
#endif
    }

#ifdef MHC_POST_NORM
    // each rooted row's hc while its rstd travels (the Sinkhorn before the norm phase: the roots' own rstd reads
    // overlap it)
    for (uint32_t i = 0; i < n_rooted; ++i) {
        hc_row();
    }

    // norm phase: per unit, y [c] (cb_y front) * its row's rstd (cb_nr) * the weight (cb_ng) -> cb_yn; DEST 0 / 1 = the
    // rstd / weight tiles (once a window), DEST 2..7 = up to 6 columns
    pack_reconfig_data_format(cb_yn);
    for (uint32_t u = start; u < start + count; ++u) {
        const mhc_stats::Unit U = mhc_stats::unit_of(u, rt, bw, ct);
        cb_wait_front(cb_nr, 1);
        cb_wait_front(cb_ng, 1);
        cb_wait_front(cb_y, bw);
        cb_reserve_back(cb_yn, bw);
        for (uint32_t c0 = 0; c0 < U.valid; c0 += 6) {
            const uint32_t c1 = U.valid < c0 + 6 ? U.valid : c0 + 6;
            tile_regs_acquire();
            reconfig_data_format_srca(cb_nr);
            copy_init(cb_nr);
            copy_from(cb_nr, 0, 0);
            copy_init(cb_ng);
            copy_from(cb_ng, 0, 1);
            reconfig_data_format_srca(cb_ng, cb_y);
            copy_init(cb_y);
            for (uint32_t c = c0; c < c1; ++c) {
                copy_from(cb_y, c, 2 + c - c0);
            }
            sfpu_init();
            for (uint32_t c = c0; c < c1; ++c) {
                switch (c) {
                    case 0: norm_apply_col<0>(); break;
                    case 1: norm_apply_col<1>(); break;
                    case 2: norm_apply_col<2>(); break;
                    case 3: norm_apply_col<3>(); break;
                    case 4: norm_apply_col<4>(); break;
                    case 5: norm_apply_col<5>(); break;
                    case 6: norm_apply_col<6>(); break;
                    default: norm_apply_col<7>(); break;
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = c0; c < c1; ++c) {
                pack_tile<true>(2 + c - c0, cb_yn, c);
            }
            tile_regs_release();
        }
        cb_push_back(cb_yn, bw);
        cb_pop_front(cb_y, bw);
        cb_pop_front(cb_ng, 1);
        cb_pop_front(cb_nr, 1);
    }
#endif
}
