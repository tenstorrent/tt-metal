// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar layer_norm, standard Welford, sliding-window two-pass compute.
//
// Threading: the kernel runs on C NEOs (num_threads = C). NEO t owns tile-rows t, t+C, ... of the
// core's NCHt rows. Every NEO runs the same number of rounds, rounds = ceil(NCHt / C): a NEO that runs out of
// rows recomputes NEO 0's row for that round (the reader feeds it the same tiles and the writer stores the
// identical result to the same pages), so all NEOs, reader harts and writer harts stay in lock-step on the
// STRIDED rings.
//
// Per tile-row:
//
//   Pass 1 (statistics): the reader streams the Wt tiles of the row through dfb::in, whose per-NEO capacity
//   is `cb_length` tiles (the window). Each tile is transposed into DST and fed to the Welford LLK (mean / M2
//   live in SFPU registers across the row). The first `Wt - tail` tiles are popped as consumed; the last
//   `tail = min(Wt, cb_length / 2)` tiles are NOT popped and stay resident in the window. The tail is capped at
//   half the window so the reader always has a double-buffer half free for the DM0 ISR credit batch. The host
//   picks cb_length / 2 as a divisor of Wt so every ring's per-row stream is a whole number of credit batches.
//
//   Pass 2 (normalize + affine): stats are finalised to mean / rstd column tiles. The resident tail is
//   normalised first (it is at the front of the FIFO), then the reader re-sends ("reloads") tiles
//   [0, Wt - tail) and those follow. Processing order for the row is
//       p = 0..Wt-1  ->  source tile index  src(p) = (Wt - tail + p) % Wt
//   and dfb::out receives output tiles in that order. The writer maps p -> src(p) when it writes, and the
//   reader streams gamma / beta tiles in that order too (one tile per processed tile, popped by compute).
//
// Ring access on Quasar: the rings shared with other NEOs are STRIDED (stride C), and the binary unpack
// (unpack_AB, used by every *_bcast_* op) indexes entries, not strides, so every binary op reads tile index 0
// with a wait(1) / op / pop(1) rhythm. The unary unpack (transpose_tile) is stride-aware, so pass 1 may index
// the resident tail by j. Packing a batch is stride-aware.
//
// Reader contract (dfb::in, per row):   push tiles 0..Wt-1, then push tiles 0..Wt-tail-1 again.
// Reader contract (dfb::eps):           push 1 bf16 column-scalar tile of eps per NEO, once per kernel.
// Reader contract (dfb::gamma / beta):  per row, Wt tiles in processing order (if FUSE_GAMMA / FUSE_BETA).
// Writer contract (dfb::out):           per row, Wt tiles arrive in processing order.
//
// Compile-time args: Wt, W, cb_length, block_size, compute_threads   Runtime args: NCHt
// Defines:           FUSE_GAMMA, FUSE_BETA (gate the dfb::gamma / dfb::beta bindings)
//
// Notes:
//  * The Welford LLK is used without a reciprocal LUT (reciprocal_size = 0 -> float division).
//  * transpose_tile on Quasar does not support fp32 unpack-to-dest (tt-llk#1559), so dfb::in must be a
//    16-bit format on Quasar for now.
//  * On Quasar the pack destination is baked at pack_init; every pack-target switch re-inits.

#include <array>
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/welford.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t cb_length = get_arg(args::cb_length);  // dfb::in capacity per NEO, in tiles
    constexpr uint32_t blk = get_arg(args::block_size);       // pass-2 block (tiles per DST acquire)
    constexpr uint32_t C = get_arg(args::compute_threads);    // NEOs sharing this node

    constexpr uint32_t tile_width = 32;
    static_assert(Wt > 0 && cb_length > 0 && blk > 0 && C > 0, "bad geometry");
    static_assert(cb_length % 2 == 0, "window must split into two ISR credit halves");
    static_assert(blk <= cb_length / 2, "pass-2 block must fit in one half of the window");
    static_assert(blk <= 4, "pass-2 block is limited by DST capacity (fp32 acc)");

    // Window tail that stays resident between the passes: at most half the window.
    constexpr uint32_t tail = Wt < cb_length / 2 ? Wt : cb_length / 2;
    constexpr uint32_t head = Wt - tail;
    // Valid columns in the last tile of the row (rows after the transpose).
    constexpr uint32_t last_tile_rows = (W % tile_width) == 0 ? tile_width : W % tile_width;

    constexpr auto dfb_in = dfb::in;
    constexpr auto dfb_eps = dfb::eps;
    constexpr auto dfb_ex = dfb::ex;        // mean, column-oriented
    constexpr auto dfb_ex2 = dfb::ex2;      // var, column-oriented
    constexpr auto dfb_ex2pe = dfb::ex2pe;  // rstd = 1/sqrt(var + eps), column-oriented
    constexpr auto dfb_xmm = dfb::xmm;      // normalise / affine intermediate
    constexpr auto dfb_out = dfb::out;
#ifdef FUSE_GAMMA
    constexpr bool do_gamma = true;
    constexpr auto dfb_gamma = dfb::gamma;
#else
    constexpr bool do_gamma = false;
    constexpr auto dfb_gamma = dfb_in;  // unused
#endif
#ifdef FUSE_BETA
    constexpr bool do_beta = true;
    constexpr auto dfb_beta = dfb::beta;
#else
    constexpr bool do_beta = false;
    constexpr auto dfb_beta = dfb_in;  // unused
#endif

    // One object per buffer for the whole kernel (Quasar drains on ~DataflowBuffer, so no locals).
    DataflowBuffer in(dfb_in);
    DataflowBuffer eps(dfb_eps);
    DataflowBuffer ex(dfb_ex);
    DataflowBuffer ex2(dfb_ex2);
    DataflowBuffer ex2pe(dfb_ex2pe);
    DataflowBuffer xmm(dfb_xmm);
    DataflowBuffer out(dfb_out);
    DataflowBuffer gamma(dfb_gamma);
    DataflowBuffer beta(dfb_beta);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // No reciprocal LUT: the LLK divides.
    static constexpr std::array<uint32_t, 0> no_lut{};

    // Every NEO runs the same number of rounds (see header); C == 1 gives NCHt rounds.
    const uint32_t rounds = (NCHt + C - 1) / C;

    compute_kernel_hw_startup(dfb_in, dfb_ex);
    eps.wait_front(1);  // pushed once by the reader, popped after its last use

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

    for (uint32_t round = 0; round < rounds; ++round) {
        // ------------------------------------------------------------------ pass 1: Welford stats
        reconfig_data_format_srca(dfb_in);
        transpose_init(dfb_in);
        welford_init();  // ClearStats: fresh mean / M2 for this row

        tile_regs_acquire();
        uint32_t n = 0;  // samples consumed so far along W (indexes the LLK's 1/(n+1))

        // Head: consume and pop.
        for (uint32_t wt = 0; wt < head; ++wt) {
            in.wait_front(1);
            transpose_tile(dfb_in, 0, input_dst);
            welford_update<0>(input_dst, n, no_lut);
            n += tile_width;
            in.pop_front(1);
        }
        // Tail: consume but keep resident for pass 2 (unary unpack indexing is stride-aware).
        for (uint32_t j = 0; j < tail; ++j) {
            const uint32_t wt = head + j;
            in.wait_front(static_cast<uint16_t>(j + 1));
            transpose_tile(dfb_in, j, input_dst);
            const uint32_t rows = (wt == Wt - 1) ? last_tile_rows : tile_width;
            if (rows == tile_width) {
                welford_update<0>(input_dst, n, no_lut);
            } else {
                welford_update_rows<0>(input_dst, n, 0, rows, no_lut);
            }
            n += rows;
        }

        // mean / var -> row 0 of mean_dst / var_dst (scale 1/W), then rotate into column 0 so the
        // pass-2 FPU ops can column-broadcast them.
        welford_finalize_to_row<0>(mean_dst, W - 1, no_lut);
#ifdef LN_DEBUG_STATS_FROM_EPS
        // TEMPORARY sim bring-up: mean = var = eps (column 0 of the eps tile), already column-oriented.
        reconfig_data_format_srca(dfb_in, dfb_eps);
        copy_init(dfb_eps);
        copy_tile(dfb_eps, 0, mean_dst);
        copy_tile(dfb_eps, 0, var_dst);
#else
        transpose_dest_init<DST_ACCUM_MODE>(dfb_in);
        transpose_dest<DST_ACCUM_MODE>(mean_dst);
        transpose_dest<DST_ACCUM_MODE>(var_dst);
#endif
        tile_regs_commit();

        ex.reserve_back(1);
        ex2.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(dfb_ex);
        pack_init(dfb_ex);
        pack_tile(mean_dst, dfb_ex);
        pack_reconfig_data_format(dfb_ex2);
        pack_init(dfb_ex2);
        pack_tile(var_dst, dfb_ex2);
        tile_regs_release();
        ex.push_back(1);
        ex2.push_back(1);

        // rstd = rsqrt(var + eps)
        ex2.wait_front(1);
        tile_regs_acquire();
        reconfig_data_format(dfb_ex2, dfb_eps);
        add_init(dfb_ex2, dfb_eps);
        add_tiles(dfb_ex2, dfb_eps, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();
        ex2.pop_front(1);
        if (round == rounds - 1) {
            eps.pop_front(1);  // last use: release it before the pass-2 waits so the reader's drain can finish
        }
        pack_batch(dfb_ex2pe, ex2pe, 1);

        // ------------------------------------------------------------ pass 2: normalise + affine
        // FIFO order is [tail tiles..., reloaded head tiles...]; blocks just walk it, one entry at a time.

        ex.wait_front(1);
        ex2pe.wait_front(1);

        for (uint32_t p = 0; p < Wt; p += blk) {
            const uint32_t nb = (Wt - p) < blk ? (Wt - p) : blk;

            // xmm = x - mean
            tile_regs_acquire();
            reconfig_data_format(dfb_in, dfb_ex);
            sub_bcast_cols_init(dfb_in, dfb_ex);
            for (uint32_t i = 0; i < nb; ++i) {
                in.wait_front(1);
                sub_tiles_bcast_cols(dfb_in, dfb_ex, 0, 0, i);
                in.pop_front(1);
            }
            tile_regs_commit();
            pack_batch(dfb_xmm, xmm, nb);

            // norm = xmm * rstd   -> xmm again if gamma/beta follow, else straight to out
            {
                constexpr auto dfb_dst = (do_gamma || do_beta) ? dfb_xmm : dfb_out;
                auto& dst_obj = [&]() -> auto& {
                    if constexpr (do_gamma || do_beta) {
                        return xmm;
                    } else {
                        return out;
                    }
                }();
                tile_regs_acquire();
                reconfig_data_format(dfb_xmm, dfb_ex2pe);
                mul_bcast_cols_init(dfb_xmm, dfb_ex2pe);
                for (uint32_t i = 0; i < nb; ++i) {
                    xmm.wait_front(1);
                    mul_tiles_bcast_cols(dfb_xmm, dfb_ex2pe, 0, 0, i);
                    xmm.pop_front(1);
                }
                tile_regs_commit();
                pack_batch(dfb_dst, dst_obj, nb);
            }

            // * gamma  (row-broadcast; gamma tiles arrive in processing order)
            if constexpr (do_gamma) {
                constexpr auto dfb_dst = do_beta ? dfb_xmm : dfb_out;
                auto& dst_obj = [&]() -> auto& {
                    if constexpr (do_beta) {
                        return xmm;
                    } else {
                        return out;
                    }
                }();
                tile_regs_acquire();
                reconfig_data_format(dfb_xmm, dfb_gamma);
                mul_bcast_rows_init(dfb_xmm, dfb_gamma);
                for (uint32_t i = 0; i < nb; ++i) {
                    xmm.wait_front(1);
                    gamma.wait_front(1);
                    mul_tiles_bcast_rows(dfb_xmm, dfb_gamma, 0, 0, i);
                    xmm.pop_front(1);
                    gamma.pop_front(1);
                }
                tile_regs_commit();
                pack_batch(dfb_dst, dst_obj, nb);
            }

            // + beta  (row-broadcast; beta tiles arrive in processing order)
            if constexpr (do_beta) {
                tile_regs_acquire();
                reconfig_data_format(dfb_xmm, dfb_beta);
                add_bcast_rows_init(dfb_xmm, dfb_beta);
                for (uint32_t i = 0; i < nb; ++i) {
                    xmm.wait_front(1);
                    beta.wait_front(1);
                    add_tiles_bcast_rows(dfb_xmm, dfb_beta, 0, 0, i);
                    xmm.pop_front(1);
                    beta.pop_front(1);
                }
                tile_regs_commit();
                pack_batch(dfb_out, out, nb);
            }
        }

        ex.pop_front(1);
        ex2pe.pop_front(1);
    }
}
