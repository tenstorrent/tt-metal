// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre compute kernel. Per block (block_token_tiles token tile-rows of this rank's K slice):
//
//   w_split_block       ONCE, before block 0, fp32 W only: each fp32 W tile k of cb_weight is rewritten
//                        in place (aliased cb_weight_split) into the exact bf16 pair [W_hi(k), W_lo(k)]
//   project_block       bf16 W:  matmul_block helper: mix partial = X_blk @ W_slice -> cb_partial [mix rows]
//                        (in0 WaitAndRetainOnLastBlock, num_k_blocks = 1: the X block stays resident)
//                        fp32 W:  project_block_pieces: X_blk @ W_hi + X_blk @ W_lo in ONE DEST window
//   sumsq_block         eltwise_chain Mul x*x DEST-accumulated over K        -> cb_sq_acc
//                        + reduce<SUM, REDUCE_ROW, Accurate>                  -> cb_partial [sumsq rows]
//   combine_block       root only: rank-ordered fp32 SFPU fold of cb_gathered -> cb_combined
//   coefficients_block  custom SFPU block op: r, pre, post, logits          -> cb_coef_out (+cb_logits_coef)
//   ymix_block          eltwise_chain Mul x_i * bcast_col(pre_i), DEST-accumulated over i -> cb_y_out
//   sinkhorn_block      custom SFPU block op on the owned rows              -> cb_comb_coef
//
// Raw-LLK deviations (helper considered and rejected; see op_design.md "Helpers considered and rejected"):
//   * project_block_pieces (fp32 W): matmul_block cannot accumulate two in1 operands (W_hi, W_lo) into one
//     DEST window — a second call would pack and FPU-reload the fp32 partial (tf32 truncation). Realized
//     as a thin block op over matmul_tiles: per output sub-block, all K x pieces products accumulate in
//     DEST; the X block is waited but never popped (same retention contract as the helper path).
//   * w_split_block: copy_tile (UnpackToDestFp32) -> SFPU bit mask / subtract -> two packs per tile; the
//     chain has one pack terminal per element, and the in-place alias rewrite needs explicit page indices.
//   * combine_block: reduce<AccumulateViaAdd> reads the fp32 partials through the FPU (tf32 truncation);
//     eltwise_chain cannot express a runtime group_cores-deep fold in one DEST window. Realized with the
//     chain's own two primitives, copy_tile (UnpackToDestFp32) + add_binary_tile (SFPU), in a loop.
//   * coefficients_block / sinkhorn_block: the math runs across slots WITHIN one tile (row / column sums
//     over the n x n matrix, per-token softmax); chain SFPU elements are whole-tile elementwise. Realized as
//     custom SFPI functions over one DEST tile that call the non-approximate SFPI exp primitive, with a
//     Newton-refined reciprocal / rsqrt (no bare approximate reciprocal).

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
constexpr uint32_t cb_bias_coef = get_compile_time_arg_val(2);
constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(3);
constexpr uint32_t cb_sq_acc = get_compile_time_arg_val(4);
constexpr uint32_t cb_partial = get_compile_time_arg_val(5);
constexpr uint32_t cb_gathered = get_compile_time_arg_val(6);
constexpr uint32_t cb_combined = get_compile_time_arg_val(7);
constexpr uint32_t cb_coef_in = get_compile_time_arg_val(8);
constexpr uint32_t cb_coef_out = get_compile_time_arg_val(9);
constexpr uint32_t cb_logits_coef = get_compile_time_arg_val(10);
constexpr uint32_t cb_comb_coef = get_compile_time_arg_val(11);
constexpr uint32_t cb_pre_cols = get_compile_time_arg_val(12);
constexpr uint32_t cb_y_out = get_compile_time_arg_val(13);
constexpr uint32_t n_streams = get_compile_time_arg_val(14);
constexpr uint32_t block_token_tiles = get_compile_time_arg_val(15);
constexpr uint32_t core_k_tiles_max = get_compile_time_arg_val(16);
constexpr uint32_t group_cores = get_compile_time_arg_val(17);
constexpr uint32_t cb_weight_split = get_compile_time_arg_val(18);
constexpr uint32_t w_pieces = get_compile_time_arg_val(19);       // 1: bf16 W (exact); 2: fp32 W -> hi/lo
constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(20);  // reader's W push quantum
// Math fidelity of the W_lo products (pieces p >= 1). W_lo is tiny (<= 2^-8 |W|) and, for a tf32-valued W,
// carries <= 3 significant bits, so a low fidelity costs ~nothing in precision and saves FPU phases.
constexpr auto w_lo_fidelity = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(21));
// == MATH_FIDELITY, but visible on every TRISC (the unpack order depends on whether the lo pass is separate).
constexpr auto w_main_fidelity = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(22));
constexpr uint32_t cb_w_matmul = w_pieces > 1 ? cb_weight_split : cb_weight;
// fp32 X (x_pieces == 2): exact projection / sum x^2 path (Refinement 2). cb_x_fp32 aliases cb_x_resident
// (same allocation, UnpackToDestFp32); cb_x_pieces streams one K chunk window of bf16 [x_hi, x_mid] pieces;
// cb_mix_run carries the fp32 running mix between chunk windows.
constexpr uint32_t cb_x_fp32 = get_compile_time_arg_val(23);
constexpr uint32_t cb_x_pieces = get_compile_time_arg_val(24);
constexpr uint32_t cb_mix_run = get_compile_time_arg_val(25);
constexpr uint32_t x_pieces = get_compile_time_arg_val(26);         // 1: bf16 X (exact); 2: fp32 X -> hi/mid
constexpr uint32_t x_chunk_k_tiles = get_compile_time_arg_val(27);  // K tiles per piece window
constexpr uint32_t x_piece_rows = get_compile_time_arg_val(28);     // piece window rows (>= sub-block height)
constexpr uint32_t x_window_pages = x_pieces * x_chunk_k_tiles * x_piece_rows;  // nominal push per window

#ifdef TRISC_MATH
namespace mhc_sfpu {

using namespace sfpi;
using ckernel::sfpu::Converter;

constexpr int N = static_cast<int>(n_streams);
constexpr int MIX = N * (N + 2);  // slots 0..MIX-1 hold the mixes, slot MIX holds sum(x^2)
constexpr int LOGIT0 = 2 * N;     // first logit / comb slot
constexpr int SCRATCH0 = 31;      // unused slots (MIX+1 .. 31) serve as per-lane scratch
constexpr int SCRATCH1 = 30;
constexpr int TILE_SLOTS = 32;  // dst_reg stride of one DEST tile

// 1/x for x > 0 finite: hardware seed + two Newton steps (~fp32 accurate).
sfpi_inline vFloat recip_pos(vFloat x) {
    vFloat y = approx_recip(x);
    vFloat t = 2.0f - x * y;
    y = y * t;
    t = 2.0f - x * y;
    y = y * t;
    return y;
}

// 1/sqrt(x) for x > 0 finite: bit-trick seed + four Newton steps.
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

// w_split_block on one DEST tile pair: tile 0 = fp32 W in, -> tile 0 = W_hi (bf16-truncated), tile 1 =
// W_lo = W - W_hi (exact in fp32; exactly bf16 for a tf32-valued W).
void w_split_hi_lo() {
    vUInt hi_mask = 0xFFFF0000;
#pragma GCC unroll 8
    for (int k = 0; k < TILE_SLOTS; ++k) {
        vFloat v = dst_reg[k];
        vFloat hi = as<vFloat>(as<vUInt>(v) & hi_mask);
        dst_reg[k] = hi;
        dst_reg[TILE_SLOTS + k] = v - hi;
    }
}

// x split on one DEST tile pair: tile 0 = fp32 x in -> tile 0 = x_hi = bf16-truncate(x), tile 1 = x_mid =
// bf16-RNE(x - x_hi). x - x_hi is exact in fp32; the dropped residual is <= 2^-17 |x| and unbiased.
void x_split_hi_mid() {
    vUInt hi_mask = 0xFFFF0000;
#pragma GCC unroll 8
    for (int k = 0; k < TILE_SLOTS; ++k) {
        vFloat v = dst_reg[k];
        vFloat hi = as<vFloat>(as<vUInt>(v) & hi_mask);
        vUInt u = as<vUInt>(v - hi);
        vUInt lsb = (u << 15) >> 31;
        u = u + 0x7FFF;
        u = u + lsb;
        u = u & hi_mask;
        dst_reg[k] = hi;
        dst_reg[TILE_SLOTS + k] = as<vFloat>(u);
    }
}

// Exact fp32 sum of squares: DEST tile 0 (+)= sum_{j < count} (DEST tile 1+j)^2, SFPU fp32 arithmetic.
template <int COUNT, bool FIRST>
void sq_accumulate() {
#pragma GCC unroll 4
    for (int k = 0; k < TILE_SLOTS; ++k) {
        vFloat v0 = dst_reg[TILE_SLOTS + k];
        vFloat acc = v0 * v0;
        if constexpr (!FIRST) {
            acc = dst_reg[k] + acc;
        }
#pragma GCC unroll 8
        for (int j = 1; j < COUNT; ++j) {
            vFloat v = dst_reg[TILE_SLOTS * (1 + j) + k];
            acc = acc + v * v;
        }
        dst_reg[k] = acc;
    }
}

// Round-to-nearest-even to tf32 (10 explicit mantissa bits): the FPU then reads the value losslessly.
sfpi_inline vFloat rne_tf32(vFloat v) {
    vUInt u = as<vUInt>(v);
    vUInt lsb = (u << 18) >> 31;
    u = u + 0xFFF;
    u = u + lsb;
    u = (u >> 13) << 13;
    return as<vFloat>(u);
}

// coefficients_block on one coefficient-major tile: DEST tile 0 = raw sums, DEST tile 1 = bias.
void coefficients(
    uint32_t a_pre_bits,
    uint32_t a_post_bits,
    uint32_t a_res_bits,
    uint32_t eps_bits,
    uint32_t norm_eps_bits,
    uint32_t inv_nc_bits) {
    {
        vFloat ssq = dst_reg[MIX];
        vFloat inv_nc = Converter::as_float(inv_nc_bits);
        vFloat neps = Converter::as_float(norm_eps_bits);
        vFloat r = rsqrt_pos(ssq * inv_nc + neps);
        dst_reg[MIX] = r;
    }
#pragma GCC unroll 32
    for (int k = 0; k < MIX; ++k) {
        const uint32_t a_bits = k < N ? a_pre_bits : (k < 2 * N ? a_post_bits : a_res_bits);
        vFloat r = dst_reg[MIX];
        vFloat a = Converter::as_float(a_bits);
        vFloat z = dst_reg[k] * r;
        z = z * a + dst_reg[TILE_SLOTS + k];
        if (k < N) {
            vFloat s = sigmoid_acc(z);
            vFloat e = Converter::as_float(eps_bits);
            dst_reg[k] = rne_tf32(s + e);
        } else if (k < 2 * N) {
            vFloat s = sigmoid_acc(z);
            dst_reg[k] = s + s;
        } else {
            dst_reg[k] = z;
        }
    }
}

// Column normalisation: m[i][j] *= 1 / (sum_i m[i][j] + eps), sums in index order.
sfpi_inline void col_norm(uint32_t eps_bits) {
#pragma GCC unroll 8
    for (int j = 0; j < N; ++j) {
        vFloat s = dst_reg[LOGIT0 + j];
        for (int i = 1; i < N; ++i) {
            s = s + dst_reg[LOGIT0 + i * N + j];
        }
        vFloat e = Converter::as_float(eps_bits);
        vFloat rc = recip_pos(s + e);
        for (int i = 0; i < N; ++i) {
            dst_reg[LOGIT0 + i * N + j] = dst_reg[LOGIT0 + i * N + j] * rc;
        }
    }
}

// Row normalisation: m[i][j] *= 1 / (sum_j m[i][j] + eps).
sfpi_inline void row_norm(uint32_t eps_bits) {
#pragma GCC unroll 8
    for (int i = 0; i < N; ++i) {
        vFloat s = dst_reg[LOGIT0 + i * N];
        for (int j = 1; j < N; ++j) {
            s = s + dst_reg[LOGIT0 + i * N + j];
        }
        vFloat e = Converter::as_float(eps_bits);
        vFloat rr = recip_pos(s + e);
        for (int j = 0; j < N; ++j) {
            dst_reg[LOGIT0 + i * N + j] = dst_reg[LOGIT0 + i * N + j] * rr;
        }
    }
}

// sinkhorn_block on one coefficient-major tile (DEST tile 0): logits -> comb, all iterations in DEST.
void sinkhorn(uint32_t eps_bits, uint32_t iters) {
    // m = softmax_j(L) + eps, row max subtracted (overflow-safe).
#pragma GCC unroll 8
    for (int i = 0; i < N; ++i) {
        const int row = LOGIT0 + i * N;
        {
            vFloat mx = dst_reg[row];
            for (int j = 1; j < N; ++j) {
                vFloat t = dst_reg[row + j];
                v_if(t > mx) { mx = t; }
                v_endif;
            }
            dst_reg[SCRATCH0] = mx;
            dst_reg[SCRATCH1] = 0.0f;
        }
        for (int j = 0; j < N; ++j) {
            vFloat lv = dst_reg[row + j];
            vFloat mxv = dst_reg[SCRATCH0];
            vFloat d = lv - mxv;
            vFloat ex = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(d);
            dst_reg[row + j] = ex;
            dst_reg[SCRATCH1] = dst_reg[SCRATCH1] + ex;
        }
        {
            vFloat rs = recip_pos(dst_reg[SCRATCH1]);
            vFloat e = Converter::as_float(eps_bits);
            for (int j = 0; j < N; ++j) {
                dst_reg[row + j] = dst_reg[row + j] * rs + e;
            }
        }
    }
    col_norm(eps_bits);
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        row_norm(eps_bits);
        col_norm(eps_bits);
    }
}

}  // namespace mhc_sfpu
#endif

// One-time SFPU init for the custom block ops (config reg + addr mods + counters); no programmable
// constants are used by the custom ops.
ALWI void custom_sfpu_init() { MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

// combine_block (root): S[j] = sum_rank gathered[rank][j], fixed rank order, fp32 SFPU adds.
ALWI void combine_block() {
    constexpr uint32_t slot_tiles = 2 * block_token_tiles;
    cb_wait_front(cb_gathered, group_cores * slot_tiles);
    cb_reserve_back(cb_combined, slot_tiles);
    reconfig_data_format_srca(cb_gathered);
    pack_reconfig_data_format(cb_combined);
    copy_tile_to_dst_init_short(cb_gathered);
    add_binary_tile_init();
    for (uint32_t j = 0; j < slot_tiles; ++j) {
        tile_regs_acquire();
        copy_tile(cb_gathered, j, 0);
        for (uint32_t r = 1; r < group_cores; ++r) {
            copy_tile(cb_gathered, r * slot_tiles + j, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_combined);
        tile_regs_release();
    }
    cb_push_back(cb_combined, slot_tiles);
    cb_pop_front(cb_gathered, group_cores * slot_tiles);
}

// w_split_block (once, before block 0): cb_weight fp32 tile k (bytes [4096k, 4096k+4096)) -> cb_weight_split
// bf16 pages 2k (W_hi) and 2k+1 (W_lo), the SAME bytes. In place is safe: tile k is fully unpacked into DEST
// before its pair is packed, and later unpacks only touch tiles > k.
ALWI void w_split_block(uint32_t core_k_tiles) {
    constexpr uint32_t pair_limit = compute_kernel_lib::DEST_AUTO_LIMIT / 2;  // one DEST pair per W tile
    constexpr uint32_t tiles_per_window = pair_limit < w_chunk_tiles ? pair_limit : w_chunk_tiles;
    static_assert(w_chunk_tiles % tiles_per_window == 0, "a DEST window must not straddle a W chunk");
    cb_reserve_back(cb_weight_split, 2 * core_k_tiles);
    reconfig_data_format_srca(cb_weight);
    pack_reconfig_data_format(cb_weight_split);
    copy_tile_to_dst_init_short(cb_weight);
    custom_sfpu_init();
    for (uint32_t k0 = 0; k0 < core_k_tiles; k0 += tiles_per_window) {
        const uint32_t nt = (core_k_tiles - k0) < tiles_per_window ? (core_k_tiles - k0) : tiles_per_window;
        if (k0 % w_chunk_tiles == 0) {
            // W arrives in chunks; waits are cumulative (cb_weight is never popped).
            const uint32_t upto = k0 + w_chunk_tiles;
            cb_wait_front(cb_weight, upto < core_k_tiles ? upto : core_k_tiles);
        }
        tile_regs_acquire();
        for (uint32_t j = 0; j < nt; ++j) {
            copy_tile(cb_weight, k0 + j, 2 * j);
            MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::w_split_hi_lo, 2 * j, VectorMode::None)));
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < 2 * nt; ++j) {
            pack_tile<true>(j, cb_weight_split, 2 * k0 + j);
        }
        tile_regs_release();
    }
    cb_push_back(cb_weight_split, 2 * core_k_tiles);
    cb_wait_front(cb_weight_split, 2 * core_k_tiles);  // resident for the whole kernel, never popped
}

// project_block_pieces: mix[t] = sum_k sum_p X[t][k] @ Wp[k] (Wp page k*PIECES + p), every product of an
// output sub-block accumulated in one DEST window, then packed once (fp32) to cb_partial. Piece 0 (W_hi, or
// the bf16 W itself) runs at MATH_FIDELITY; pieces >= 1 (W_lo) at w_lo_fidelity (math re-init only; same DEST).
template <uint32_t PIECES>
ALWI void project_block_pieces(uint32_t extent, uint32_t core_k_tiles, uint32_t sb_h) {
    cb_wait_front(cb_x_resident, extent * core_k_tiles);  // retained: sumsq + y-mix reuse the block
    reconfig_data_format(cb_w_matmul, cb_x_resident);     // matmul: srca = in1, srcb = in0
    pack_reconfig_data_format(cb_partial);
    matmul_init(cb_x_resident, cb_w_matmul);
    for (uint32_t r0 = 0; r0 < extent; r0 += sb_h) {
        constexpr bool lo_reinit = PIECES > 1 && w_lo_fidelity != w_main_fidelity;
        tile_regs_acquire();
        if constexpr (lo_reinit) {
            if (r0 > 0) {
                MATH((llk_math_matmul_init<w_main_fidelity, MM_THROTTLE>(cb_x_resident, cb_w_matmul)));
            }
        }
        for (uint32_t k = 0; k < core_k_tiles; ++k) {
            for (uint32_t r = 0; r < sb_h; ++r) {
                const uint32_t x_idx = (r0 + r) * core_k_tiles + k;
                matmul_tiles(cb_x_resident, cb_w_matmul, x_idx, k * PIECES, r);
                if constexpr (!lo_reinit) {
                    for (uint32_t p = 1; p < PIECES; ++p) {
                        matmul_tiles(cb_x_resident, cb_w_matmul, x_idx, k * PIECES + p, r);
                    }
                }
            }
        }
        if constexpr (lo_reinit) {
            MATH((llk_math_matmul_init<w_lo_fidelity, MM_THROTTLE>(cb_x_resident, cb_w_matmul)));
            for (uint32_t k = 0; k < core_k_tiles; ++k) {
                for (uint32_t r = 0; r < sb_h; ++r) {
                    const uint32_t x_idx = (r0 + r) * core_k_tiles + k;
                    for (uint32_t p = 1; p < PIECES; ++p) {
                        UNPACK((llk_unpack_AB_matmul(cb_x_resident, cb_w_matmul, x_idx, k * PIECES + p)));
                        MATH((llk_math_matmul<w_lo_fidelity, MM_THROTTLE>(r)));
                    }
                }
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_partial, sb_h);
        tile_regs_wait();
        for (uint32_t r = 0; r < sb_h; ++r) {
            pack_tile(r, cb_partial);
        }
        tile_regs_release();
        cb_push_back(cb_partial, sb_h);
    }
}

// x_split_window: the bf16 pieces of X rows [r0, r0 + rows) x K tiles [k0, k0 + kc) -> one cb_x_pieces
// window, page (r * kc + kk) * x_pieces + q. Recomputed per K chunk from the resident fp32 block (read via
// the UnpackToDestFp32 alias), so the pieces never exist as a second resident copy of the block.
ALWI void x_split_window(uint32_t r0, uint32_t rows, uint32_t k0, uint32_t kc, uint32_t core_k_tiles) {
    constexpr uint32_t tiles_per_dest = compute_kernel_lib::DEST_AUTO_LIMIT / x_pieces;
    cb_reserve_back(cb_x_pieces, x_window_pages);
    reconfig_data_format_srca(cb_x_fp32);
    pack_reconfig_data_format(cb_x_pieces);
    copy_tile_to_dst_init_short(cb_x_fp32);
    custom_sfpu_init();
    const uint32_t total = rows * kc;
    for (uint32_t u0 = 0; u0 < total; u0 += tiles_per_dest) {
        const uint32_t nt = (total - u0) < tiles_per_dest ? (total - u0) : tiles_per_dest;
        tile_regs_acquire();
        for (uint32_t j = 0; j < nt; ++j) {
            const uint32_t u = u0 + j;
            const uint32_t r = u / kc;
            const uint32_t kk = u - r * kc;
            copy_tile(cb_x_fp32, (r0 + r) * core_k_tiles + k0 + kk, x_pieces * j);
            MATH((_llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::x_split_hi_mid, x_pieces * j, VectorMode::None)));
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < x_pieces * nt; ++j) {
            pack_tile<true>(j, cb_x_pieces, x_pieces * u0 + j);
        }
        tile_regs_release();
    }
    cb_push_back(cb_x_pieces, x_window_pages);
}

// project_block_split (fp32 X): mix[t] = sum_k sum_{q + p <= 1} x_q[t][k] @ W_p[k], streamed over K in
// x_chunk_k_tiles windows. Per (sub-block, chunk): x_split_window, then ONE DEST window that reloads the fp32
// running partial exactly (UnpackToDestFp32 copy; an FPU reload would truncate it), accumulates the chunk's
// products on top (main-fidelity products first, then the W_lo products at w_lo_fidelity), and packs the new
// running partial (cb_mix_run) or, on the last chunk, the mix partial (cb_partial). The dropped products
// (x_mid @ W_lo, <= 2^-16 relative) are below the FPU accumulation floor.
ALWI void project_block_split(uint32_t extent, uint32_t core_k_tiles, uint32_t sb_h) {
    static_assert(x_pieces == 2, "the fp32-X split path is hi/mid");
    constexpr uint32_t sb_max = block_token_tiles < compute_kernel_lib::DEST_AUTO_LIMIT
                                    ? block_token_tiles
                                    : compute_kernel_lib::DEST_AUTO_LIMIT;
    static_assert(sb_max <= x_piece_rows, "cb_x_pieces / cb_mix_run must hold a full projection sub-block");
    constexpr bool lo_reinit = w_pieces > 1 && w_lo_fidelity != w_main_fidelity;
    for (uint32_t r0 = 0; r0 < extent; r0 += sb_h) {
        for (uint32_t k0 = 0; k0 < core_k_tiles; k0 += x_chunk_k_tiles) {
            const uint32_t kc = (core_k_tiles - k0) < x_chunk_k_tiles ? (core_k_tiles - k0) : x_chunk_k_tiles;
            const bool first = k0 == 0;
            const bool last = k0 + kc >= core_k_tiles;
            x_split_window(r0, sb_h, k0, kc, core_k_tiles);

            cb_wait_front(cb_x_pieces, x_window_pages);
            tile_regs_acquire();
            if (!first) {
                cb_wait_front(cb_mix_run, x_piece_rows);
                reconfig_data_format_srca(cb_mix_run);
                copy_tile_to_dst_init_short(cb_mix_run);
                for (uint32_t r = 0; r < sb_h; ++r) {
                    copy_tile(cb_mix_run, r, r);
                }
            }
            reconfig_data_format(cb_w_matmul, cb_x_pieces);  // matmul: srca = in1, srcb = in0
            matmul_init(cb_x_pieces, cb_w_matmul);
            // main fidelity: x_q @ W_0 for every piece q (and every W piece if the lo pass is not separate)
            for (uint32_t kk = 0; kk < kc; ++kk) {
                for (uint32_t r = 0; r < sb_h; ++r) {
                    for (uint32_t q = 0; q < x_pieces; ++q) {
                        const uint32_t x_idx = (r * kc + kk) * x_pieces + q;
                        matmul_tiles(cb_x_pieces, cb_w_matmul, x_idx, (k0 + kk) * w_pieces, r);
                        if constexpr (!lo_reinit) {
                            for (uint32_t p = 1; p + q < 2 && p < w_pieces; ++p) {
                                matmul_tiles(cb_x_pieces, cb_w_matmul, x_idx, (k0 + kk) * w_pieces + p, r);
                            }
                        }
                    }
                }
            }
            if constexpr (lo_reinit) {
                // x_hi @ W_lo (q = 0, p = 1) at the lo fidelity
                MATH((llk_math_matmul_init<w_lo_fidelity, MM_THROTTLE>(cb_x_pieces, cb_w_matmul)));
                for (uint32_t kk = 0; kk < kc; ++kk) {
                    for (uint32_t r = 0; r < sb_h; ++r) {
                        const uint32_t x_idx = (r * kc + kk) * x_pieces;
                        UNPACK((llk_unpack_AB_matmul(cb_x_pieces, cb_w_matmul, x_idx, (k0 + kk) * w_pieces + 1)));
                        MATH((llk_math_matmul<w_lo_fidelity, MM_THROTTLE>(r)));
                    }
                }
            }
            tile_regs_commit();
            cb_pop_front(cb_x_pieces, x_window_pages);
            if (!first) {
                cb_pop_front(cb_mix_run, x_piece_rows);
            }
            const uint32_t cb_dst = last ? cb_partial : cb_mix_run;
            const uint32_t dst_pages = last ? sb_h : x_piece_rows;  // cb_mix_run: nominal (never wraps mid-window)
            cb_reserve_back(cb_dst, dst_pages);
            pack_reconfig_data_format(cb_dst);
            tile_regs_wait();
            for (uint32_t r = 0; r < sb_h; ++r) {
                pack_tile<true>(r, cb_dst, r);
            }
            tile_regs_release();
            cb_push_back(cb_dst, dst_pages);
        }
    }
}

#ifdef TRISC_MATH
template <int C>
ALWI void sq_accumulate_run(bool first) {
    if (first) {
        _llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::sq_accumulate<C, true>, 0, VectorMode::None);
    } else {
        _llk_math_eltwise_unary_sfpu_params_(mhc_sfpu::sq_accumulate<C, false>, 0, VectorMode::None);
    }
}
// runtime tile count (ragged last round) -> compile-time SFPU body
template <uint32_t MAX_COUNT, int C = 1>
ALWI void sq_accumulate_dispatch(uint32_t count, bool first) {
    if constexpr (C < static_cast<int>(MAX_COUNT)) {
        if (count != static_cast<uint32_t>(C)) {
            sq_accumulate_dispatch<MAX_COUNT, C + 1>(count, first);
            return;
        }
    }
    sq_accumulate_run<C>(first);
}
#endif

// sumsq_row_exact (fp32 X): Q = sum_k x_k * x_k for one token tile-row, exact fp32 on the SFPU from the
// UnpackToDestFp32 alias (the FPU would read x as ~tf32), DEST-accumulated in one window -> cb_sq_acc.
ALWI void sumsq_row_exact(uint32_t base, uint32_t core_k_tiles) {
    constexpr uint32_t per_round = compute_kernel_lib::DEST_AUTO_LIMIT - 1;  // DEST tile 0 = accumulator
    static_assert(per_round >= 1 && per_round <= 7, "sq_accumulate dispatch covers 1..7 tiles per round");
    cb_reserve_back(cb_sq_acc, 1);
    reconfig_data_format_srca(cb_x_fp32);
    pack_reconfig_data_format(cb_sq_acc);
    copy_tile_to_dst_init_short(cb_x_fp32);
    custom_sfpu_init();
    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < core_k_tiles; k0 += per_round) {
        const uint32_t nt = (core_k_tiles - k0) < per_round ? (core_k_tiles - k0) : per_round;
        for (uint32_t j = 0; j < nt; ++j) {
            copy_tile(cb_x_fp32, base + k0 + j, 1 + j);
        }
        MATH((sq_accumulate_dispatch<per_round>(nt, k0 == 0)));
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sq_acc);
    tile_regs_release();
    cb_push_back(cb_sq_acc, 1);
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t core_token_tiles = get_arg_val<uint32_t>(1);
    const uint32_t core_c_tiles = get_arg_val<uint32_t>(2);
    const uint32_t rank = get_arg_val<uint32_t>(3);
    const uint32_t a_pre_bits = get_arg_val<uint32_t>(4);
    const uint32_t a_post_bits = get_arg_val<uint32_t>(5);
    const uint32_t a_res_bits = get_arg_val<uint32_t>(6);
    const uint32_t eps_bits = get_arg_val<uint32_t>(7);
    const uint32_t norm_eps_bits = get_arg_val<uint32_t>(8);
    const uint32_t inv_nc_bits = get_arg_val<uint32_t>(9);
    const uint32_t sinkhorn_iters = get_arg_val<uint32_t>(10);

    using namespace compute_kernel_lib;

    const uint32_t core_k_tiles = n_streams * core_c_tiles;
    constexpr uint32_t x_block_pages = block_token_tiles * core_k_tiles_max;  // nominal (matches reader)

    CircularBuffer x_buf(cb_x_resident);
    CircularBuffer w_buf(cb_weight);
    CircularBuffer partial_buf(cb_partial);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_matmul, cb_partial);

    if constexpr (w_pieces > 1) {
        w_split_block(core_k_tiles);
    }

    // Resident constant: the bias tile is waited once and never popped.
    cb_wait_front(cb_bias_coef, 1);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        const uint32_t extent =
            (core_token_tiles - row0) < block_token_tiles ? (core_token_tiles - row0) : block_token_tiles;

        // ---- project_block: mix partial = X_blk @ W_slice (out subblock height = largest divisor <= DEST) ----
        uint32_t sb_h = 1;
        for (uint32_t h = DEST_AUTO_LIMIT; h > 1; --h) {
            if (extent % h == 0) {
                sb_h = h;
                break;
            }
        }
        if constexpr (x_pieces > 1) {
            // cb_x_fp32 tracks cb_x_resident page for page (compute is its producer and consumer; the
            // reader's data is guaranteed by the cb_x_resident wait).
            cb_reserve_back(cb_x_fp32, x_block_pages);
            cb_push_back(cb_x_fp32, x_block_pages);
            cb_wait_front(cb_x_resident, extent * core_k_tiles);
            cb_wait_front(cb_x_fp32, x_block_pages);
            project_block_split(extent, core_k_tiles, sb_h);
        } else if constexpr (w_pieces > 1) {
            project_block_pieces<w_pieces>(extent, core_k_tiles, sb_h);
        } else {
            matmul_block<
                false,
                false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndRetainOnLastBlock,
                InputPolicy::WaitAndRetainOnLastBlock>(
                x_buf,
                w_buf,
                partial_buf,
                partial_buf,
                MatmulBlockShape::of(extent / sb_h, 1, sb_h, 1, core_k_tiles, 1));
        }

        // ---- sumsq_block: per row, Q = sum_k x_k*x_k (DEST-accumulated), then fp32 row-collapse ----
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t base = t * core_k_tiles;
            if constexpr (x_pieces > 1) {
                sumsq_row_exact(base, core_k_tiles);
            } else {
                eltwise_chain(
                    IterationShape::tiles(core_k_tiles),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(
                            cb_x_resident,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        input(
                            cb_x_resident,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        Dst::D0,
                        DestAccumulation::WholeShape>{base, base},
                    PackTile<output(
                        cb_sq_acc,
                        ReservePolicy::OneUpfront,
                        PushPolicy::OneAtEnd,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Direct,
                        DestAccumulation::WholeShape)>{});
            }
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sq_acc,
                cb_reduce_scaler,
                cb_partial,
                ReduceInputPolicy::WaitAndPopPerTile,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Accurate>(ReduceInputBlockShape::single());
        }

        // ---- combine_block (root only) ----
        if (rank == 0) {
            combine_block();
        }

        // ---- coefficients_block ----
        cb_wait_front(cb_coef_in, extent);
        cb_reserve_back(cb_coef_out, extent);
        reconfig_data_format_srca(cb_coef_in);
        pack_reconfig_data_format(cb_coef_out);
        copy_tile_to_dst_init_short(cb_coef_in);
        custom_sfpu_init();
        for (uint32_t t = 0; t < extent; ++t) {
            const bool owned = ((row0 + t) % group_cores) == rank;
            if (owned) {
                cb_reserve_back(cb_logits_coef, 1);
            }
            tile_regs_acquire();
            copy_tile(cb_coef_in, t, 0);
            copy_tile(cb_bias_coef, 0, 1);
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                mhc_sfpu::coefficients,
                0,
                VectorMode::None,
                a_pre_bits,
                a_post_bits,
                a_res_bits,
                eps_bits,
                norm_eps_bits,
                inv_nc_bits)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_coef_out);
            if (owned) {
                pack_tile(0, cb_logits_coef);
            }
            tile_regs_release();
            if (owned) {
                cb_push_back(cb_logits_coef, 1);
            }
        }
        cb_push_back(cb_coef_out, extent);
        cb_pop_front(cb_coef_in, extent);

        // ---- ymix_block: y[c] = sum_i x[i][c] * bcast_col(pre_i), n-deep DEST accumulation per output ----
        cb_wait_front(cb_pre_cols, n_streams * extent);
        for (uint32_t t = 0; t < extent; ++t) {
            eltwise_chain(
                IterationShape::grid(core_c_tiles, n_streams),
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(
                        cb_x_resident,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    input(
                        input(
                            cb_pre_cols,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Row,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        BroadcastDim::Col),
                    Dst::D0,
                    DestAccumulation::PerRow>{t * core_k_tiles, t * n_streams},
                PackTile<output(
                    cb_y_out,
                    ReservePolicy::PerOuter,
                    PushPolicy::PerOuter,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Direct,
                    DestAccumulation::PerRow)>{});
        }
        cb_pop_front(cb_pre_cols, n_streams * extent);
        if constexpr (x_pieces > 1) {
            cb_pop_front(cb_x_fp32, x_block_pages);  // alias kept in lockstep
        }
        cb_pop_front(cb_x_resident, x_block_pages);  // X block freed: the reader may load block+2

        // ---- sinkhorn_block (owned rows), after the y-mix (stall-shadow reorder) ----
        bool sinkhorn_initialized = false;
        for (uint32_t t = 0; t < extent; ++t) {
            if (((row0 + t) % group_cores) != rank) {
                continue;
            }
            if (!sinkhorn_initialized) {
                reconfig_data_format_srca(cb_logits_coef);
                pack_reconfig_data_format(cb_comb_coef);
                copy_tile_to_dst_init_short(cb_logits_coef);
                custom_sfpu_init();
                sinkhorn_initialized = true;
            }
            cb_wait_front(cb_logits_coef, 1);
            cb_reserve_back(cb_comb_coef, 1);
            tile_regs_acquire();
            copy_tile(cb_logits_coef, 0, 0);
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                mhc_sfpu::sinkhorn, 0, VectorMode::None, eps_bits, sinkhorn_iters)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_comb_coef);
            tile_regs_release();
            cb_push_back(cb_comb_coef, 1);
            cb_pop_front(cb_logits_coef, 1);
        }
    }
}
