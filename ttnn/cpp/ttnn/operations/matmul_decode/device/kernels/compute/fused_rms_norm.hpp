// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Fused adaRMS-norm PROLOGUE for the partial-width-sharded matmul_decode computes.
//
//   A_normed = (A * rsqrt(mean(A^2) + eps)) * weight [+ bias]      (weight/bias per CHANNEL over K)
//
// WHY: the pi0.5 denoise block is DISPATCH-bound, not compute-bound -- ~31% of its traced wall-clock is
// inter-op overhead (~3.7 us per op across ~10 ops). Normalizing inside the consumer matmul removes TWO
// ops per site: the standalone rms_norm AND the InterleavedToSharded that fed it (the norm wants a
// block-sharded input while the residual stream is interleaved). ~12 us removed per site.
//
// WHY IT IS POSSIBLE HERE: reader_partial_width_sharded.cpp calls gather_full_a(), so the ENTIRE A row
// is already resident on every compute core. sum(A^2) over K therefore needs NO cross-core reduction --
// exactly the cost that made the kv_sdpa split-KV experiment lose.
//
// KEY EFFICIENCY POINT -- only this core's K-SLICE is normalized. Every core holds the full gathered A,
// but phase1_partial only ever READS this core's Kc_tiles slice, so normalizing all K_tiles would be
// ~4x wasted work (that naive version measured out at ~7 us/op and made the fusion break even). Only
// the sum(A^2) needs the full row. So:
//     square + reduce over all K_tiles   (unavoidable: the statistic spans the row)
//     scale/weight/bias over Kc_tiles    (this core's slice only)
// ~88 tile-ops per core instead of ~128, and the weight/bias reads shrink 4x with it.
//
// The normalized tiles are written into a FULL-SIZE out_cb at the SAME scattered indices phase1_partial
// computes from k_global, so phase1's indexing is untouched and the caller just passes out_cb in place
// of full_in0. Slots outside this core's slice are never written and never read.
//
// weight/bias are broadcast DOWN the rows (bcast_rows) so the host does NOT pre-replicate them --
// scale1/shift are recomputed every denoise step, so a host-side replicate would add back an op.

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// CB contract (counts in A tiles):
//   in0_cb    : M_tiles * K_tiles, gathered A (sender-block order; read-only here)
//   sq_cb     : M_tiles * K_tiles scratch for A^2 (consumed by the reduce)
//   scaler_cb : 1 tile, 1/(K_tiles*TILE_WIDTH) packed bf16, produced by the reader
//   stat_cb   : M_tiles tiles, rsqrt(mean+eps) per row (value in column 0)
//   w_cb      : Kc_tiles tiles, THIS CORE's slice of the per-channel weight
//   b_cb      : Kc_tiles tiles, this core's slice of the bias; only touched when HAS_BIAS
//   t1_cb     : M_tiles * Kc_tiles staging for A*stat (compact slice indices)
//   t2_cb     : M_tiles * Kc_tiles staging for t1*weight; unused when !HAS_BIAS
//   out_cb    : M_tiles * K_tiles, normalized A (only this core's slice is written)
// NOTE: every CB id is a TEMPLATE parameter, not a function argument. compute_kernel_lib::reduce takes
// its buffer ids as template args, so they must be constant expressions -- a function parameter is not
// one. They are compile-time args on the caller side anyway.
template <
    uint32_t M_tiles,
    uint32_t K_tiles,
    uint32_t Kc_tiles,
    uint32_t inA_K_tiles_per_core,
    uint32_t sender_slice_tiles,
    bool HAS_BIAS,
    uint32_t eps_bits,
    uint32_t in0_cb,
    uint32_t sq_cb,
    uint32_t scaler_cb,
    uint32_t stat_cb,
    uint32_t w_cb,
    uint32_t b_cb,
    uint32_t t1_cb,
    uint32_t t2_cb,
    uint32_t out_cb>
inline void fused_rms_norm_prologue(uint32_t k_offset) {
    using namespace ckernel;
    constexpr uint32_t num_tiles = M_tiles * K_tiles;

    // ---- 1. sq = A * A over the WHOLE row (the statistic spans K) ---------------------------------
    // A separate pass because compute_kernel_lib::reduce sums x, not x^2 (layernorm_sharded.cpp too).
    // BATCHED over DST: one tile_regs_acquire/commit/wait/release cycle per DST_BATCH tiles, not per
    // tile. The per-tile version cost ~7.5 us for this prologue -- 32 full DST cycles here alone -- which
    // was more than the ops the fusion removes. Protocol overhead, not math.
    constexpr uint32_t DST_BATCH = 8;  // DST capacity in half-sync bf16 mode
    reconfig_data_format(in0_cb, in0_cb);
    pack_reconfig_data_format(sq_cb);
    mul_tiles_init(in0_cb, in0_cb);
    cb_reserve_back(sq_cb, num_tiles);
    for (uint32_t base = 0; base < num_tiles; base += DST_BATCH) {
        const uint32_t n = (num_tiles - base < DST_BATCH) ? (num_tiles - base) : DST_BATCH;
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            mul_tiles(in0_cb, in0_cb, base + j, base + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(j, sq_cb);
        }
        tile_regs_release();
    }
    cb_push_back(sq_cb, num_tiles);

    // ---- 2. stat = rsqrt(mean(A^2) + eps) --------------------------------------------------------
    // REDUCE_ROW over [M_tiles x K_tiles] yields M_tiles tiles, each with its row's sum in column 0.
    // scaler_cb carries 1/N so the sum arrives already divided. +eps and rsqrt are folded in through
    // PostReduceOp, which runs while the result is still in DST -- one pass saved. The default
    // WaitAndPopPerTile policy consumes sq_cb, so do NOT pop it again.
    // Only the first five template args are named: the trailing policy/reconfig/fp32 params keep their
    // defaults (the default input policy is WaitAndPopPerTile, which consumes sq_cb for us) and
    // PostReduceOp is DEDUCED from the lambda below -- spelling those types out does not compile from
    // kernel context anyway.
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, sq_cb, scaler_cb, stat_cb>(
        compute_kernel_lib::ReduceInputBlockShape::of(M_tiles, K_tiles, 1),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
        [](uint32_t dst_idx) {
            add_unary_tile(dst_idx, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(dst_idx);
        });

    // ---- 3/4/5. normalize THIS CORE's K-slice ----------------------------------------------------
    // Three passes rather than one fused block: these APIs take operands from CBs, not DST, so the
    // scale (per-row, bcast along COLUMNS) / weight / bias (per-channel, bcast down ROWS) chain cannot
    // share a tile_regs block.
    //
    // Each pass is its own loop so the format reconfig happens ONCE per pass, and each stages through a
    // separate compact CB rather than mutating one in place: after cb_push_back the write pointer has
    // advanced, so a pack_tile(.., idx) "in-place" update would not land where the unpacker reads. The
    // existing gated-residual epilogue in this file stages the same way (reduce -> mm -> mmg -> out).
    //
    // t1_cb / t2_cb are only Kc_tiles deep (this core's slice, compact indices 0..Kc_tiles-1); only the
    // final pass scatters to out_cb at the sender-block index phase1_partial will read.
    cb_wait_front(w_cb, Kc_tiles);
    if constexpr (HAS_BIAS) {
        cb_wait_front(b_cb, Kc_tiles);
    }

    // Pass 3: t1[kc] = A[i] * stat[m]
    reconfig_data_format(in0_cb, stat_cb);
    pack_reconfig_data_format(t1_cb);
    mul_bcast_cols_init_short(in0_cb, stat_cb);
    cb_reserve_back(t1_cb, M_tiles * Kc_tiles);
    for (uint32_t m = 0; m < M_tiles; ++m) {
        for (uint32_t base = 0; base < Kc_tiles; base += DST_BATCH) {
            const uint32_t n = (Kc_tiles - base < DST_BATCH) ? (Kc_tiles - base) : DST_BATCH;
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t k_global = k_offset + base + j;
                const uint32_t sender = k_global / inA_K_tiles_per_core;
                const uint32_t kc_local = k_global - sender * inA_K_tiles_per_core;
                // The tile address phase1_partial reads for this (m, k_global).
                const uint32_t i = sender * sender_slice_tiles + kc_local + m * inA_K_tiles_per_core;
                mul_tiles_bcast_cols(in0_cb, stat_cb, i, m, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, t1_cb);
            }
            tile_regs_release();
        }
    }
    cb_push_back(t1_cb, M_tiles * Kc_tiles);
    cb_pop_front(stat_cb, M_tiles);

    // Pass 4: t2[kc] = t1[kc] * weight[kc]   (straight to out_cb when there is no bias)
    constexpr uint32_t p4_dst = HAS_BIAS ? t2_cb : out_cb;
    cb_wait_front(t1_cb, M_tiles * Kc_tiles);
    reconfig_data_format(t1_cb, w_cb);
    pack_reconfig_data_format(p4_dst);
    mul_bcast_rows_init_short(t1_cb, w_cb);
    if constexpr (HAS_BIAS) {
        cb_reserve_back(t2_cb, M_tiles * Kc_tiles);
    } else {
        cb_reserve_back(out_cb, num_tiles);
    }
    for (uint32_t m = 0; m < M_tiles; ++m) {
        for (uint32_t base = 0; base < Kc_tiles; base += DST_BATCH) {
            const uint32_t n = (Kc_tiles - base < DST_BATCH) ? (Kc_tiles - base) : DST_BATCH;
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                mul_tiles_bcast_rows(t1_cb, w_cb, m * Kc_tiles + base + j, base + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                if constexpr (HAS_BIAS) {
                    pack_tile(j, t2_cb);
                } else {
                    const uint32_t k_global = k_offset + base + j;
                    const uint32_t sender = k_global / inA_K_tiles_per_core;
                    const uint32_t kc_local = k_global - sender * inA_K_tiles_per_core;
                    pack_tile(j, out_cb, sender * sender_slice_tiles + kc_local + m * inA_K_tiles_per_core);
                }
            }
            tile_regs_release();
        }
    }
    cb_pop_front(t1_cb, M_tiles * Kc_tiles);
    if constexpr (!HAS_BIAS) {
        cb_push_back(out_cb, num_tiles);
        return;
    }
    cb_push_back(t2_cb, M_tiles * Kc_tiles);

    // Pass 5: out[i] = t2[kc] + bias[kc], scattered into the sender-block layout
    cb_wait_front(t2_cb, M_tiles * Kc_tiles);
    reconfig_data_format(t2_cb, b_cb);
    pack_reconfig_data_format(out_cb);
    add_bcast_rows_init_short(t2_cb, b_cb);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t m = 0; m < M_tiles; ++m) {
        for (uint32_t base = 0; base < Kc_tiles; base += DST_BATCH) {
            const uint32_t n = (Kc_tiles - base < DST_BATCH) ? (Kc_tiles - base) : DST_BATCH;
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                add_tiles_bcast_rows(t2_cb, b_cb, m * Kc_tiles + base + j, base + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t k_global = k_offset + base + j;
                const uint32_t sender = k_global / inA_K_tiles_per_core;
                const uint32_t kc_local = k_global - sender * inA_K_tiles_per_core;
                pack_tile(j, out_cb, sender * sender_slice_tiles + kc_local + m * inA_K_tiles_per_core);
            }
            tile_regs_release();
        }
    }
    cb_pop_front(t2_cb, M_tiles * Kc_tiles);
    cb_push_back(out_cb, num_tiles);
}
