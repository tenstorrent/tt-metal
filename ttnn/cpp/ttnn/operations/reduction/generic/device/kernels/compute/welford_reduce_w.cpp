// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================================
// REWRITE NOTE (replaces online Welford with a shifted two-pass algorithm):
//   mean     = shift + avg(x - shift)
//   variance = avg((x - mean)^2)          [avg divisor: W if !correction, W-1 if correction]
//
// This algorithm needs the W-dimension read TWICE (once to get `mean`, once to get variance
// around that mean). Welford's online recurrence exists specifically to avoid that second read,
// so removing it is not a pure in-kernel algorithm swap. Two things live OUTSIDE this file and
// MUST change for this kernel to be correct:
//
//   (1) READER: must push each row's `Wt` input tiles to dfb::in TWICE per NCHt iteration
//       (2*Wt tiles total per row instead of Wt). dfb::in's double-buffering depth
//       (input_tiles_per_cb = 2 in welford_reduce_program_factory.cpp) does not need to change,
//       it's still simple streaming -- only the total tile count the reader emits per row does.
//
//   (2) PROGRAM FACTORY: if W is not a multiple of tile_width, the last tile's padded columns
//       are zero in DRAM. For a plain reduce, zero-padding is harmless (adds 0 to a sum). For
//       variance it is NOT harmless: (0 - mean)^2 is generally nonzero and would corrupt the
//       result. This file guards the tail with mask_tile(), which needs a `dfb::mask` buffer
//       (one tile, 1s for valid columns / 0s for padding on the last tile) provided by the
//       reader/program factory. Until that buffer exists, WELFORD_TAIL_MASKED must stay
//       undefined and W must be a multiple of tile_width for correct results.
//
// This file has NOT been validated against real hardware. Before switching production traffic
// off the online-Welford path, diff this kernel's output against welford_reduce_w.cpp (the file
// it replaces) on real inputs, including inputs that exercise the ragged-W tail.
// ============================================================================================

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reduce.h"
#include "api/compute/compute_kernel_hw_startup.h"

#ifdef WELFORD_TAIL_MASKED
#include "api/compute/mask.h"
#endif

#ifdef WELFORD_POST_MUL
// SFPU multiply-by-scalar (mul_unary_tile) applied to the reduced output. See issue #45222.
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Runtime args:
    // Total number of outer-loop iterations (N * C * Ht),
    // i.e. how many independent row-reductions this core must perform.
    const uint32_t NCHt = get_arg(args::NCHt);

    // Compile-time args:
    // Number of tiles along the W (reduction) dimension.
    constexpr auto Wt = get_arg(args::Wt);
    // The actual number of elements along W (before tiling).
    constexpr auto W = get_arg(args::W);
    // Number of elements per tile in the W dimension
    // (typically 32, but can be smaller for narrow tiles).
    constexpr auto tile_width = get_arg(args::tile_width);
#ifdef WELFORD_POST_MUL
    // Packed fp32 post-multiplier applied to the reduced output via mul_unary_tile (SFPU).
    // For var this is scalar^2, for std it is |scalar| (see welford_reduce_program_factory).
    constexpr auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif
    // Whether to apply Bessel's correction (divide by N-1 instead of N).
    constexpr bool correction = get_arg(args::correction) != 0;
    // Whether to compute standard deviation (sqrt of variance) instead of variance.
    constexpr bool is_std = get_arg(args::is_std) != 0;

    constexpr uint32_t onetile = 1;

    // Number of valid columns in the last tile along W. When W is not a multiple of
    // tile_width, columns [last_tile_cols, tile_width) of the last tile are zero-padding
    // and must not contribute to the variance sum (see mask note above).
    constexpr uint32_t last_tile_cols = ((W % tile_width) == 0) ? tile_width : (W % tile_width);
    static_assert(
        last_tile_cols == tile_width,
        "Ragged W (W % tile_width != 0) requires WELFORD_TAIL_MASKED plus a reader-provided "
        "dfb::mask buffer -- see the REWRITE NOTE at the top of this file. Define "
        "WELFORD_TAIL_MASKED and remove this static_assert once that buffer is wired up.");

    // Divisors for the two averages. Mean is always the plain population average over W.
    // Variance's divisor depends on Bessel's correction.
    constexpr float inv_W = 1.0f / static_cast<float>(W);
    constexpr float inv_var_N = correction ? (1.0f / static_cast<float>(W - 1)) : inv_W;
    constexpr uint32_t inv_W_bits = std::bit_cast<uint32_t>(inv_W);
    constexpr uint32_t inv_var_N_bits = std::bit_cast<uint32_t>(inv_var_N);

    // Destination register indices inside the Tensix DST register file.
    constexpr uint32_t work_dst = 0;    // scratch: broadcast shift/mean, diff, diff^2, accumulator staging
    constexpr uint32_t accum_dst = 1;   // running sum(x - shift) in pass 1, sum((x-mean)^2) in pass 2
    constexpr uint32_t result_dst = 2;  // final mean / variance / std

    // Buffer that the reader fills with input tiles. Per the REWRITE NOTE, the reader must push
    // each row's Wt tiles TWICE (pass 1, then pass 2) -- 2*Wt tiles per NCHt iteration.
    DataflowBuffer dfb_in(dfb::in);
    // Standard reduce scaling-factor tile (already filled with 1.0 by the shared reader infra
    // for every reduce-family kernel; see welford_reduce_program_factory.cpp -- SCALAR_DFB is
    // populated regardless of which compute kernel consumes it). We reuse it here as the SUM
    // scaler for reduce_tile: our own divide-by-W(-1) is applied afterwards as a compile-time
    // SFPU multiply, exactly like WELFORD_POST_MUL does for the user scalar.
    DataflowBuffer dfb_scalar(dfb::scalar);
    // Buffer where the final output tile is written for the writer kernel to consume.
    DataflowBuffer dfb_out(dfb::out);
    // Scratch bounce buffer: DST results are packed here and re-unpacked with column
    // broadcast (BroadcastType::COL) to materialize a full 32-wide broadcast tile from a
    // single per-row scalar living in column 0. Also used as the running-accumulator CB
    // between successive reduce_tile calls (mirrors the dfb_acc pattern in reduce_w_neg.cpp:
    // reduce_tile accumulates into whatever is already resident in its DST slot, so the prior
    // partial sum is copied back into DST before each new reduce_tile call).
    DataflowBuffer dfb_var(dfb::var);
    // Second scratch buffer: holds the per-tile (x - shift) / (x - mean) difference so it can
    // be squared via mul_tiles (which needs two CB operands) and so sub_tiles has two CB
    // operands (broadcast tile is only available materialized in a CB, not in DST).
    DataflowBuffer dfb_diff(dfb::diff);
#ifdef WELFORD_TAIL_MASKED
    // One tile, reader-filled: 1.0 in valid columns [0, last_tile_cols), 0.0 in the padded
    // tail columns of the last tile. Only consumed when processing the last Wt tile of a row.
    DataflowBuffer dfb_mask(dfb::mask);
#endif

    compute_kernel_hw_startup(dfb::in, dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // ------------------------------------------------------------------------------
        // Extract the per-row shift: x[row, 0] of the row's first tile, broadcast across
        // all `tile_width` columns via BroadcastType::COL. Using an element of the data
        // itself as the shift (rather than e.g. 0) is what keeps sum(x - shift) numerically
        // small and stable, which is the entire point of the "shifted" two-pass algorithm.
        // ------------------------------------------------------------------------------
        dfb_in.wait_front(onetile);  // peek row's tile 0; not popped here, pass 1 consumes it below

        tile_regs_acquire();
        unary_bcast_init<BroadcastType::COL>(dfb::in);
        unary_bcast<BroadcastType::COL>(dfb::in, 0, work_dst);
        tile_regs_commit();
        tile_regs_wait();
        dfb_var.reserve_back(onetile);
        pack_reconfig_data_format(dfb::var);
        pack_tile(work_dst, dfb::var);
        tile_regs_release();
        dfb_var.push_back(onetile);

        // ------------------------------------------------------------------------------
        // Pass 1: mean = shift + avg(x - shift)
        // ------------------------------------------------------------------------------
        dfb_var.wait_front(onetile);  // shift, fully broadcast, now resident as a CB tile
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            if (wt > 0) {
                dfb_in.wait_front(onetile);
            }

            // diff = x - shift
            tile_regs_acquire();
            sub_init(dfb::in, dfb::var);
            sub_tiles(dfb::in, dfb::var, 0, 0, work_dst);
            tile_regs_commit();
            tile_regs_wait();
            dfb_diff.reserve_back(onetile);
            pack_reconfig_data_format(dfb::diff);
            pack_tile(work_dst, dfb::diff);
            tile_regs_release();
            dfb_diff.push_back(onetile);
            dfb_diff.wait_front(onetile);

            // accum += reduce_row_sum(diff)   (accumulate-into-DST, mirrors reduce_w_neg.cpp)
            tile_regs_acquire();
            if (wt > 0) {
                dfb_var.wait_front(onetile);
                copy_init(dfb::var);
                copy_tile(dfb::var, 0, accum_dst);
            }
            constexpr bool swap_operands = true;  // REDUCE_ROW SUM/AVG swaps operands; see reduce.h
            if constexpr (swap_operands) {
                reconfig_data_format(dfb::scalar, dfb::diff);
            }
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::diff, dfb::scalar, dfb::var);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::diff, dfb::scalar, 0, 0, accum_dst);
            reduce_uninit();
            tile_regs_wait();
            dfb_diff.pop_front(onetile);
            dfb_in.pop_front(onetile);
            if (wt > 0) {
                dfb_var.pop_front(onetile);
            }
            dfb_var.reserve_back(onetile);
            tile_regs_commit();
            pack_reconfig_data_format(dfb::var);
            pack_tile(accum_dst, dfb::var);
            tile_regs_release();
            dfb_var.push_back(onetile);
        }  // wt (pass 1)

        // mean = shift + sum(x - shift) * (1/W)
        dfb_var.wait_front(onetile);
        tile_regs_acquire();
        copy_init(dfb::var);
        copy_tile(dfb::var, 0, result_dst);  // sum(x - shift)
        binop_with_scalar_tile_init();
        mul_unary_tile(result_dst, inv_W_bits);  // avg(x - shift)
        unary_bcast_init<BroadcastType::COL>(dfb::var);
        // shift is still sitting, fully broadcast, in dfb_var's PREVIOUS entry -- but that
        // entry was already reserved/pushed over by the pass-1 accumulator above, so we
        // re-derive shift the same way it was produced: broadcast column 0 of the row's
        // first input tile. This re-reads tile 0 from dfb::in, which per the REWRITE NOTE
        // the reader must supply again as the first tile of pass 2's Wt tiles.
        tile_regs_commit();
        tile_regs_wait();
        dfb_var.pop_front(onetile);
        dfb_diff.reserve_back(onetile);  // stash avg(x - shift) while we fetch shift again
        pack_reconfig_data_format(dfb::diff);
        pack_tile(result_dst, dfb::diff);
        tile_regs_release();
        dfb_diff.push_back(onetile);

        dfb_in.wait_front(onetile);  // pass 2's tile 0 (re-sent by the reader)
        tile_regs_acquire();
        unary_bcast_init<BroadcastType::COL>(dfb::in);
        unary_bcast<BroadcastType::COL>(dfb::in, 0, work_dst);  // shift again
        tile_regs_commit();
        tile_regs_wait();
        dfb_var.reserve_back(onetile);
        pack_reconfig_data_format(dfb::var);
        pack_tile(work_dst, dfb::var);  // shift, broadcast, back in a CB
        tile_regs_release();
        dfb_var.push_back(onetile);
        dfb_var.wait_front(onetile);
        dfb_diff.wait_front(onetile);  // avg(x - shift), stashed above

        tile_regs_acquire();
        add_init(dfb::diff, dfb::var);
        add_tiles(dfb::diff, dfb::var, 0, 0, result_dst);  // mean = avg(x-shift) + shift
        tile_regs_commit();
        tile_regs_wait();
        dfb_diff.pop_front(onetile);
        dfb_var.reserve_back(onetile);
        pack_reconfig_data_format(dfb::var);
        pack_tile(result_dst, dfb::var);  // mean, one row-0 scalar per row
        tile_regs_release();
        dfb_var.push_back(onetile);

        // Broadcast mean across all tile_width columns so it can be used with sub_tiles below.
        dfb_var.wait_front(onetile);
        tile_regs_acquire();
        unary_bcast_init<BroadcastType::COL>(dfb::var);
        unary_bcast<BroadcastType::COL>(dfb::var, 0, work_dst);
        tile_regs_commit();
        tile_regs_wait();
        dfb_var.pop_front(onetile);
        dfb_var.reserve_back(onetile);
        pack_reconfig_data_format(dfb::var);
        pack_tile(work_dst, dfb::var);  // mean, fully broadcast
        tile_regs_release();
        dfb_var.push_back(onetile);
        dfb_var.wait_front(onetile);  // mean_bcast, resident as a CB tile for all of pass 2

        // ------------------------------------------------------------------------------
        // Pass 2: variance = avg((x - mean)^2)   [re-reads the row's Wt tiles]
        // ------------------------------------------------------------------------------
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            dfb_in.wait_front(onetile);

            // diff = x - mean
            tile_regs_acquire();
            sub_init(dfb::in, dfb::var);
            sub_tiles(dfb::in, dfb::var, 0, 0, work_dst);
#ifdef WELFORD_TAIL_MASKED
            if (wt == (Wt - 1)) {
                // Zero the padded tail columns of the last tile so they don't pollute the
                // variance sum: (0 - mean)^2 would otherwise be spuriously nonzero.
                dfb_mask.wait_front(onetile);
                mask_tile_init();
                mask_tile(work_dst, work_dst + 1);
            }
#else
            static_assert(
                Wt == 0 || last_tile_cols == tile_width,
                "unreachable: guarded by the file-level static_assert above");
#endif
            tile_regs_commit();
            tile_regs_wait();
            dfb_diff.reserve_back(onetile);
            pack_reconfig_data_format(dfb::diff);
            pack_tile(work_dst, dfb::diff);
            tile_regs_release();
            dfb_diff.push_back(onetile);
            dfb_diff.wait_front(onetile);
#ifdef WELFORD_TAIL_MASKED
            if (wt == (Wt - 1)) {
                dfb_mask.pop_front(onetile);
            }
#endif

            // sq = diff * diff
            tile_regs_acquire();
            mul_init(dfb::diff, dfb::diff);
            mul_tiles(dfb::diff, dfb::diff, 0, 0, work_dst);
            tile_regs_commit();
            tile_regs_wait();
            dfb_diff.pop_front(onetile);
            dfb_diff.reserve_back(onetile);
            pack_reconfig_data_format(dfb::diff);
            pack_tile(work_dst, dfb::diff);
            tile_regs_release();
            dfb_diff.push_back(onetile);
            dfb_diff.wait_front(onetile);

            // accum += reduce_row_sum(sq)
            tile_regs_acquire();
            if (wt > 0) {
                dfb_var.wait_front(onetile);  // NOTE: dfb_var still holds mean_bcast at wt==0;
                                               // from wt==1 on, this slot instead holds the
                                               // running accumulator (see push below).
            }
            // ...
            constexpr bool swap_operands2 = true;
            if constexpr (swap_operands2) {
                reconfig_data_format(dfb::scalar, dfb::diff);
            }
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::diff, dfb::scalar, dfb::out);
            if (wt > 0) {
                copy_init(dfb::var);
                copy_tile(dfb::var, 0, accum_dst);
                dfb_var.pop_front(onetile);
            }
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::diff, dfb::scalar, 0, 0, accum_dst);
            reduce_uninit();
            tile_regs_wait();
            dfb_diff.pop_front(onetile);
            dfb_in.pop_front(onetile);
            dfb_var.reserve_back(onetile);
            tile_regs_commit();
            pack_reconfig_data_format(dfb::var);
            pack_tile(accum_dst, dfb::var);
            tile_regs_release();
            dfb_var.push_back(onetile);
        }  // wt (pass 2)

        // variance = sum((x-mean)^2) * (1/(W or W-1)); std = sqrt(variance)
        dfb_var.wait_front(onetile);
        tile_regs_acquire();
        copy_init(dfb::var);
        copy_tile(dfb::var, 0, result_dst);
        binop_with_scalar_tile_init();
        mul_unary_tile(result_dst, inv_var_N_bits);
        if constexpr (is_std) {
            sqrt_tile_init();
            sqrt_tile(result_dst);
        }
#ifdef WELFORD_POST_MUL
        // Apply the user scalar to the reduced output: var(s*x)=s^2 var(x), std(s*x)=|s| std(x).
        mul_unary_tile(result_dst, post_mul_scaler_bits);
#endif
        tile_regs_commit();
        dfb_var.pop_front(onetile);

        dfb_out.reserve_back(onetile);
        tile_regs_wait();
        pack_reconfig_data_format(dfb::out);
        pack_tile(result_dst, dfb::out);
        tile_regs_release();
        dfb_out.push_back(onetile);

    }  // NCHt loop
}