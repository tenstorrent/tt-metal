// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for backward_softmax (VJP of softmax).
//
// Math: grad_input = output * (grad_output - sum(output * grad_output, dim))
//
// Two-pass streaming algorithm — per lane:
//
//   Pass 1, per block b in [0, NUM_BLOCKS):
//     mul(cb_grad_output, cb_output, cb_prod, block_shape)
//     accumulate_reduce_block<SUM, reduce_dim>(cb_prod, cb_scaler, cb_sum,
//                                              reduce_block_shape, b, NUM_BLOCKS)
//     // After the final block, cb_sum holds 1 tile with the lane sum at [0,0].
//
//   Pass 2, per block b in [0, NUM_BLOCKS):
//     sub<SCALAR, A=WaitAndPopPerTile, B=WaitUpfrontNoPop>(
//         cb_grad_output, cb_sum, cb_centered, block_shape)
//     mul(cb_output, cb_centered, cb_grad_input, block_shape)
//
//   cb_pop_front(cb_sum, 1)
//
// All work goes through the kernel-lib helpers — no raw tile_regs / pack_tile
// loops in this kernel. fp32_dest_acc is set in ComputeConfigDescriptor; the
// helpers honor DEST_AUTO_LIMIT (4 tiles under fp32-acc + half-sync), so
// BLOCK_SIZE up to 8 is safe (helpers internally batch).

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
    constexpr uint32_t num_lanes = get_compile_time_arg_val(3);

    // Block shapes — orientation flips with dim:
    //   dim=-1 (W reduction): block is 1 row × BLOCK_SIZE cols of tiles.
    //   dim=-2 (H reduction): block is BLOCK_SIZE rows × 1 col of tiles.
    constexpr auto reduce_block_shape = DIM_IS_W ? ckl::ReduceInputBlockShape::of(1, BLOCK_SIZE, /*batches=*/1)
                                                 : ckl::ReduceInputBlockShape::of(BLOCK_SIZE, 1, /*batches=*/1);
    constexpr auto bin_block_shape =
        DIM_IS_W ? ckl::BinaryInputBlockShape::of(1, BLOCK_SIZE) : ckl::BinaryInputBlockShape::of(BLOCK_SIZE, 1);
    constexpr auto reduce_dim = DIM_IS_W ? ckernel::ReduceDim::REDUCE_ROW : ckernel::ReduceDim::REDUCE_COL;
    // Broadcast direction for the pass-2 sub:
    //   REDUCE_ROW (dim=-1) writes per-row sums into column 0 of cb_sum's
    //     single tile → use COL broadcast (B has shape [Ht=1, 1]; LLK reads
    //     col 0 of B and replicates horizontally inside each tile, yielding
    //     A[i, j] − B[i, 0]).
    //   REDUCE_COL (dim=-2) writes per-col sums into row 0 of cb_sum's tile
    //     → use ROW broadcast (B has shape [1, Wt=1]; LLK reads row 0 of B,
    //     replicates down → A[i, j] − B[0, j]).
    // SCALAR would only be correct if the reduce produced a single value at
    // face [0,0] (i.e. REDUCE_SCALAR). REDUCE_ROW/REDUCE_COL produce a vector.
    constexpr auto sub_bcast_dim = DIM_IS_W ? ckl::BroadcastDim::COL : ckl::BroadcastDim::ROW;

    // Hardware init — once per kernel. Pass a representative input CB
    // (cb_grad_output), the scaler CB, and the output CB. Subsequent helpers
    // reconfig as needed.
    compute_kernel_hw_startup(cb_grad_output, cb_scaler, cb_grad_input);

    for (uint32_t lane = 0; lane < num_lanes; ++lane) {
        // ---------- Pass 1 ----------
        // Per block: mul(dy, y) → cb_prod, then accumulate-reduce that block
        // into cb_sum. Helper handles last-block routing (only the last block
        // emits the final reduced tile; intermediate blocks reload + re-emit
        // via Accumulate::at(cb_sum, b)).
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            ckl::mul(cb_grad_output, cb_output, cb_prod, bin_block_shape);

            ckl::accumulate_reduce_block<ckernel::PoolType::SUM, reduce_dim>(
                cb_prod, cb_scaler, cb_sum, reduce_block_shape, b, NUM_BLOCKS);
        }

        // ---------- Pass 2 ----------
        // Per block:
        //   sub<SCALAR>(dy, cb_sum) → cb_centered    (cb_sum held with WaitUpfrontNoPop)
        //   mul(y, cb_centered) → cb_grad_input
        // cb_sum survives across all blocks — popped once at end of lane.
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            ckl::
                sub<sub_bcast_dim, ckl::BinaryInputPolicy::WaitAndPopPerTile, ckl::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_grad_output, cb_sum, cb_centered, bin_block_shape);

            ckl::mul(cb_output, cb_centered, cb_grad_input, bin_block_shape);
        }

        // cb_sum was held with WaitUpfrontNoPop across pass 2 — release it
        // here so the next lane starts with cb_sum empty.
        cb_pop_front(cb_sum, 1);
    }
}
