// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar softmax over the last dimension, streamed three-pass compute.
//
// Threading: the kernel runs on C NEOs (num_threads = C). NEO t owns tile-rows t, t+C, ... of the core's
// NCHt rows. Every NEO runs rounds = ceil(NCHt / C) rounds; a NEO that runs out of rows recomputes NEO 0's
// row for that round (the reader feeds it the same tiles and the writer stores the identical result to the
// same pages), so all NEOs, reader harts and writer harts stay in lock-step on the STRIDED rings.
//
// Per tile-row (the reader streams the row once per pass through dfb::in):
//
//   Pass 1 (NUMERIC_STABLE only): rowmax = max over the row (reduce MAX / REDUCE_ROW, column 0).
//   Pass 2: per block of block_size tiles: exps = exp(x - rowmax) -> reduce SUM / REDUCE_ROW over the block
//           -> psum (column 0); rowsum += psum. Then rsum = 1 / rowsum.
//   Pass 3: per block: exps = exp(x - rowmax); out = exps * rsum (column broadcast).
//
// Ring access: the rings shared with other NEOs are STRIDED (stride C) and the binary unpack (unpack_AB,
// used by reduce and every *_bcast_* op) indexes entries, not strides, so every binary op reads tile index 0
// with a wait(1) / op / pop(1) rhythm. Packing a batch is stride-aware.
//
// Reader contract (dfb::in):     per row, tiles 0..Wt-1, repeated once per pass (3 passes, 2 if not stable).
// Reader contract (dfb::scaler): one bf16 reduce-scaler tile (row 0 of every face = 1.0) per NEO, once.
// Writer contract (dfb::out):    per row, tiles 0..Wt-1 in order.
//
// Compile-time args: Wt, block_size, compute_threads   Runtime args: NCHt   Defines: NUMERIC_STABLE
//
// The pack destination is baked at pack_init; every pack-target switch re-inits.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t blk = get_arg(args::block_size);
    constexpr uint32_t C = get_arg(args::compute_threads);
#ifdef NUMERIC_STABLE
    constexpr bool stable = true;
#else
    constexpr bool stable = false;
#endif

    static_assert(Wt > 0 && blk > 0 && C > 0, "bad geometry");
    static_assert(blk <= 4, "block is limited by DST capacity (fp32 acc)");

    constexpr auto dfb_in = dfb::in;
    constexpr auto dfb_scaler = dfb::scaler;
    constexpr auto dfb_rowmax = dfb::rowmax;
    constexpr auto dfb_exps = dfb::exps;
    constexpr auto dfb_psum = dfb::psum;
    constexpr auto dfb_rowsum = dfb::rowsum;
    constexpr auto dfb_rsum = dfb::rsum;
    constexpr auto dfb_out = dfb::out;

    // One object per buffer for the whole kernel (the drains run in ~DataflowBuffer, so no locals).
    DataflowBuffer in(dfb_in);
    DataflowBuffer scaler(dfb_scaler);
    DataflowBuffer rowmax(dfb_rowmax);
    DataflowBuffer exps(dfb_exps);
    DataflowBuffer psum(dfb_psum);
    DataflowBuffer rowsum(dfb_rowsum);
    DataflowBuffer rsum(dfb_rsum);
    DataflowBuffer out(dfb_out);

    constexpr uint32_t dst0 = 0;
    const uint32_t rounds = (NCHt + C - 1) / C;

    compute_kernel_hw_startup(dfb_in, dfb_scaler, dfb_out);
    scaler.wait_front(1);  // pushed once by the reader, popped after its last use

    // Pack `nb` DST tiles (0..nb-1) into `dst_dfb` as one batch.
    // `auto&`: DataflowBuffer is a class template (one type per access-pattern pair).
    auto pack_batch = [&](uint32_t dst_dfb, auto& dst_obj, uint32_t nb) __attribute__((always_inline)) {
        const auto nb16 = static_cast<uint16_t>(nb);
        dst_obj.reserve_back(nb16);
        tile_regs_wait();
        pack_reconfig_data_format(dst_dfb);
        pack_init(dst_dfb);  // BFD destination is baked at pack_init, not by the format reconfig
        for (uint32_t i = 0; i < nb; ++i) {
            pack_tile(i, dst_dfb);
        }
        tile_regs_release();
        dst_obj.push_back(nb16);
    };

    // exps[0..nb) = exp(in - rowmax) (or exp(in) when not stable), consuming nb tiles of dfb::in.
    auto exp_block = [&](uint32_t nb) __attribute__((always_inline)) {
        tile_regs_acquire();
        if constexpr (stable) {
            reconfig_data_format(dfb_in, dfb_rowmax);
            sub_bcast_cols_init(dfb_in, dfb_rowmax);
            for (uint32_t i = 0; i < nb; ++i) {
                in.wait_front(1);
                sub_tiles_bcast_cols(dfb_in, dfb_rowmax, 0, 0, i);
                in.pop_front(1);
            }
        } else {
            reconfig_data_format_srca(dfb_in);
            copy_init(dfb_in);
            for (uint32_t i = 0; i < nb; ++i) {
                in.wait_front(1);
                copy_tile(dfb_in, 0, i);
                in.pop_front(1);
            }
        }
        exp_tile_init();
        for (uint32_t i = 0; i < nb; ++i) {
            exp_tile(i);
        }
        tile_regs_commit();
        pack_batch(dfb_exps, exps, nb);
    };

    for (uint32_t round = 0; round < rounds; ++round) {
        // ------------------------------------------------------------------ pass 1: row max
        if constexpr (stable) {
            tile_regs_acquire();
            reconfig_data_format(dfb_in, dfb_scaler);
            rowmax.reserve_back(1);
            pack_reconfig_data_format(dfb_rowmax);
            pack_init(dfb_rowmax);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb_in, dfb_scaler, dfb_rowmax);
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                in.wait_front(1);
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb_in, dfb_scaler, 0, 0, dst0);
                in.pop_front(1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, dfb_rowmax);
            reduce_uninit(dfb_in);
            tile_regs_release();
            rowmax.push_back(1);
            rowmax.wait_front(1);
        }

        // ------------------------------------------------------------------ pass 2: row sum
        for (uint32_t p = 0; p < Wt; p += blk) {
            const uint32_t nb = (Wt - p) < blk ? (Wt - p) : blk;
            exp_block(nb);

            // Block row-sums -> rowsum for the first block, else -> psum and rowsum += psum.
            const bool first = (p == 0);
            tile_regs_acquire();
            reconfig_data_format(dfb_exps, dfb_scaler);
            if (first) {
                rowsum.reserve_back(1);
                pack_reconfig_data_format(dfb_rowsum);
                pack_init(dfb_rowsum);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_scaler, dfb_rowsum);
            } else {
                psum.reserve_back(1);
                pack_reconfig_data_format(dfb_psum);
                pack_init(dfb_psum);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_scaler, dfb_psum);
            }
            for (uint32_t i = 0; i < nb; ++i) {
                exps.wait_front(1);
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_scaler, 0, 0, dst0);
                exps.pop_front(1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, first ? dfb_rowsum : dfb_psum);
            reduce_uninit(dfb_exps);
            tile_regs_release();
            if (first) {
                rowsum.push_back(1);
            } else {
                psum.push_back(1);
                tile_regs_acquire();
                reconfig_data_format(dfb_rowsum, dfb_psum);
                add_init(dfb_rowsum, dfb_psum);
                rowsum.wait_front(1);
                psum.wait_front(1);
                add_tiles(dfb_rowsum, dfb_psum, 0, 0, dst0);
                rowsum.pop_front(1);
                psum.pop_front(1);
                tile_regs_commit();
                pack_batch(dfb_rowsum, rowsum, 1);
            }
        }
        if (round == rounds - 1) {
            scaler.pop_front(1);  // last use: release it so the reader's drain of that ring can finish
        }

        // rsum = 1 / rowsum
        tile_regs_acquire();
        reconfig_data_format_srca(dfb_rowsum);
        copy_init(dfb_rowsum);
        rowsum.wait_front(1);
        copy_tile(dfb_rowsum, 0, dst0);
        rowsum.pop_front(1);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();
        pack_batch(dfb_rsum, rsum, 1);
        rsum.wait_front(1);

        // ------------------------------------------------------------------ pass 3: normalise
        for (uint32_t p = 0; p < Wt; p += blk) {
            const uint32_t nb = (Wt - p) < blk ? (Wt - p) : blk;
            exp_block(nb);

            tile_regs_acquire();
            reconfig_data_format(dfb_exps, dfb_rsum);
            mul_bcast_cols_init(dfb_exps, dfb_rsum);
            for (uint32_t i = 0; i < nb; ++i) {
                exps.wait_front(1);
                mul_tiles_bcast_cols(dfb_exps, dfb_rsum, 0, 0, i);
                exps.pop_front(1);
            }
            tile_regs_commit();
            pack_batch(dfb_out, out, nb);
        }

        rsum.pop_front(1);
        if constexpr (stable) {
            rowmax.pop_front(1);
        }
    }
}
