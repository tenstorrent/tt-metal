// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for backward_softmax (VJP of softmax).
//
// Math: grad_input = output * (grad_output - sum(output * grad_output, dim))
//
// Two code paths, selected by compile-time STRATEGY_IS_WHOLE_ROW
// (Refinement 2):
//
//   STRATEGY_IS_WHOLE_ROW == 1 (WHOLE_ROW_DB or WHOLE_ROW_SB):
//     The reader pushes each lane's tiles ONCE (single pass). cb_grad_output
//     and cb_output are sized to hold the whole row (or 2× row for DB). The
//     compute kernel waits upfront on the whole row in pass 1, does pass 1
//     and pass 2 over those L1-resident tiles, and pops them at the end of
//     the lane. Result: each input tile is read from DRAM exactly once per
//     output tile (vs. twice in Phase 0).
//
//     Per lane:
//       mul<WaitUpfrontNoPop,WaitUpfrontNoPop>(dy, y, cb_prod, (1, reduce_dim))
//       reduce<SUM, REDUCE_DIM>(cb_prod, cb_scaler, cb_sum, (1, reduce_dim, 1))
//       sub<COL/ROW, WaitUpfrontPopAtEnd, WaitUpfrontNoPop>(dy, cb_sum,
//                                                          cb_centered,
//                                                          (1, reduce_dim))
//       mul<WaitUpfrontPopAtEnd, WaitAndPopPerTile>(y, cb_centered,
//                                                  cb_grad_input,
//                                                  (1, reduce_dim))
//       cb_pop_front(cb_sum, 1)
//
//     A and B policies on the sub/mul flip per-dim only in the block-shape
//     orientation (rows×cols).
//
//   STRATEGY_IS_WHOLE_ROW == 0 (PER_TILE_STREAM, fallback):
//     Phase-0 behavior. Block-loop over NUM_BLOCKS blocks; per-block mul →
//     accumulate_reduce_block, then per-block sub<COL/ROW> → mul. Each
//     input tile is read from DRAM TWICE.
//
// All work goes through the kernel-lib helpers — no raw tile_regs / pack_tile
// loops in this kernel. fp32_dest_acc is set in ComputeConfigDescriptor; the
// helpers honor DEST_AUTO_LIMIT (4 tiles under fp32-acc + half-sync), so
// shape rows/cols up to any value is safe (helpers internally batch).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace {
constexpr uint32_t cb_grad_output = 0;
constexpr uint32_t cb_output = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_grad_input = 16;
constexpr uint32_t cb_prod = 24;
constexpr uint32_t cb_sum = 25;
constexpr uint32_t cb_centered = 26;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(1);
    constexpr uint32_t DIM_IS_W = get_compile_time_arg_val(2);  // 1 = dim=-1, 0 = dim=-2
    constexpr uint32_t STRATEGY_IS_WHOLE_ROW = get_compile_time_arg_val(3);
    // RT arg 0: per-core lane count under multi-core distribution.
    uint32_t num_lanes = get_arg_val<uint32_t>(0);

    constexpr uint32_t reduce_dim_tiles = NUM_BLOCKS * BLOCK_SIZE;

    // Block shapes for the PER_TILE_STREAM (Phase-0) path — orientation flips
    // with dim:
    //   dim=-1 (W reduction): block is 1 row × BLOCK_SIZE cols of tiles.
    //   dim=-2 (H reduction): block is BLOCK_SIZE rows × 1 col of tiles.
    constexpr auto reduce_block_shape = DIM_IS_W ? ckl::ReduceInputBlockShape::of(1, BLOCK_SIZE, /*batches=*/1)
                                                 : ckl::ReduceInputBlockShape::of(BLOCK_SIZE, 1, /*batches=*/1);
    constexpr auto bin_block_shape =
        DIM_IS_W ? ckl::BinaryInputBlockShape::of(1, BLOCK_SIZE) : ckl::BinaryInputBlockShape::of(BLOCK_SIZE, 1);

    // Block shapes for the WHOLE_ROW path — the "block" is the entire lane.
    constexpr auto reduce_full_shape = DIM_IS_W ? ckl::ReduceInputBlockShape::of(1, reduce_dim_tiles, /*batches=*/1)
                                                : ckl::ReduceInputBlockShape::of(reduce_dim_tiles, 1, /*batches=*/1);
    constexpr auto bin_full_shape = DIM_IS_W ? ckl::BinaryInputBlockShape::of(1, reduce_dim_tiles)
                                             : ckl::BinaryInputBlockShape::of(reduce_dim_tiles, 1);

    constexpr auto reduce_dim = DIM_IS_W ? ckernel::ReduceDim::REDUCE_ROW : ckernel::ReduceDim::REDUCE_COL;

    // Broadcast direction for the pass-2 sub:
    //   REDUCE_ROW (dim=-1) writes per-row sums into column 0 of cb_sum's
    //     single tile → use COL broadcast (B has shape [Ht=1, 1]; LLK reads
    //     col 0 of B and replicates horizontally inside each tile, yielding
    //     A[i, j] − B[i, 0]).
    //   REDUCE_COL (dim=-2) writes per-col sums into row 0 of cb_sum's tile
    //     → use ROW broadcast (B has shape [1, Wt=1]; LLK reads row 0 of B,
    //     replicates down → A[i, j] − B[0, j]).
    constexpr auto sub_bcast_dim = DIM_IS_W ? ckl::BroadcastDim::COL : ckl::BroadcastDim::ROW;

    // Hardware init — once per kernel. Pass a representative input CB
    // (cb_grad_output), the scaler CB, and the output CB. Subsequent helpers
    // reconfig as needed.
    compute_kernel_hw_startup(cb_grad_output, cb_scaler, cb_grad_input);

    for (uint32_t lane = 0; lane < num_lanes; ++lane) {
        if constexpr (STRATEGY_IS_WHOLE_ROW) {
            // ---------- WHOLE_ROW path ----------
            // Both cb_grad_output and cb_output hold the lane's complete row
            // (Wt tiles for dim=-1, Ht tiles for dim=-2). The reader pushes
            // them once; this code reads them across BOTH passes from L1.

            // Pass 1: dy * y → cb_prod, then reduce → cb_sum.
            // Both inputs use WaitUpfrontNoPop so the tiles persist for pass 2.
            ckl::mul<
                ckl::BroadcastDim::NONE,
                ckl::BinaryInputPolicy::WaitUpfrontNoPop,
                ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_grad_output, cb_output, cb_prod, bin_full_shape);

            // reduce<> consumes cb_prod (WaitAndPopPerTile default), drains all
            // reduce_dim_tiles tiles, and produces 1 output tile in cb_sum with
            // the lane sum at face [0,0] of col 0 (REDUCE_ROW) or row 0
            // (REDUCE_COL). cb_scaler is held with WaitAndPopPerTile by
            // default — but reduce<> never pops the scaler (it's marked as
            // persistent across calls in the helper).
            ckl::reduce<ckernel::PoolType::SUM, reduce_dim>(cb_prod, cb_scaler, cb_sum, reduce_full_shape);

            // Pass 2 part 1: cb_centered = dy - cb_sum (broadcast).
            // A=dy: WaitUpfrontPopAtEnd — we've now consumed dy's last reader
            // of this lane; pop it at the end so the next lane can refill.
            // B=cb_sum: WaitUpfrontNoPop — survives for the next lane's reduce
            // re-using it? No — cb_sum is per-lane. But we need it across the
            // single sub call here, so the policy difference vs PopAtEnd
            // doesn't matter functionally. NoPop matches the
            // explicit-pop-at-end-of-lane pattern (mirrors the per-tile path).
            ckl::sub<
                sub_bcast_dim,
                ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_grad_output, cb_sum, cb_centered, bin_full_shape);

            // Pass 2 part 2: cb_grad_input = y * cb_centered.
            // A=y: WaitUpfrontPopAtEnd — final reader of y for this lane.
            // B=cb_centered: WaitAndPopPerTile — streaming consumer, pops as
            // it goes. cb_centered was filled by the sub above with
            // reduce_dim_tiles tiles via PerTile output; mul drains them
            // tile-by-tile.
            ckl::mul<
                ckl::BroadcastDim::NONE,
                ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                ckl::BinaryInputPolicy::WaitAndPopPerTile>(cb_output, cb_centered, cb_grad_input, bin_full_shape);

            // cb_sum was held with WaitUpfrontNoPop in the sub above — release
            // it so the next lane starts with cb_sum empty.
            cb_pop_front(cb_sum, 1);
        } else {
            // ---------- PER_TILE_STREAM path (Phase-0 fallback) ----------
            // ---------- Pass 1 ----------
            // Per block: mul(dy, y) → cb_prod, then accumulate-reduce that
            // block into cb_sum. Helper handles last-block routing (only the
            // last block emits the final reduced tile; intermediate blocks
            // reload + re-emit via Accumulate::at(cb_sum, b)).
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                ckl::mul(cb_grad_output, cb_output, cb_prod, bin_block_shape);

                ckl::accumulate_reduce_block<ckernel::PoolType::SUM, reduce_dim>(
                    cb_prod, cb_scaler, cb_sum, reduce_block_shape, b, NUM_BLOCKS);
            }

            // ---------- Pass 2 ----------
            // Per block:
            //   sub<COL/ROW>(dy, cb_sum) → cb_centered    (cb_sum held with WaitUpfrontNoPop)
            //   mul(y, cb_centered) → cb_grad_input
            // cb_sum survives across all blocks — popped once at end of lane.
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                ckl::sub<
                    sub_bcast_dim,
                    ckl::BinaryInputPolicy::WaitAndPopPerTile,
                    ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_grad_output, cb_sum, cb_centered, bin_block_shape);

                ckl::mul(cb_output, cb_centered, cb_grad_input, bin_block_shape);
            }

            // cb_sum was held with WaitUpfrontNoPop across pass 2 — release it
            // here so the next lane starts with cb_sum empty.
            cb_pop_front(cb_sum, 1);
        }
    }
}
