// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_variance.
//
// Computes per-row population variance via the proper two-pass algorithm:
//   variance = E[(x - E[x])^2]
//
// Per-row outer loop: each iteration of `for ht` runs the full pass-1 + pass-2
// pipeline on a single row, so the helpers see Ht=1 in their BlockShapes.
// Result: cb_mean / cb_variance / cb_in / cb_centered footprint is O(BLOCK_SIZE)
// regardless of input height — no Ht term in any CB. Cost is more init/uninit
// per row, which is amortized by the actual reduce work.
//
// With scaler = 1/N built into the SUM reduce, the helpers produce:
//   Pass 1: cb_mean     = mean(x)              — streaming accumulate_reduce
//   Pass 2: cb_variance = mean((x - mean)^2)   — per-block: sub<COL> →
//                                                 square_in_place → accumulate_reduce_block
//
// COMPUTE_STD_DEV: when set, sqrt is applied as the post_op_final on the
// pass-2 last-block reduce. The streaming-reduce helper routes post_op_final
// only to the last block, so sqrt runs in DST after the final accumulation,
// before pack — no extra pass over the data, and intermediate accumulator
// values stay in variance space.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_centered = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_mean = 3;
constexpr uint32_t cb_variance = 4;
constexpr uint32_t cb_out = 16;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(3);
    constexpr bool COMPUTE_STD_DEV = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(5) != 0;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // Per-row block shapes: Ht=1 inside the helpers, since the outer loop
    // serializes over rows. cb_mean / cb_variance / cb_centered each only
    // need to hold a single row's worth of working data at any time.
    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(/*Ht=*/1, BLOCK_SIZE, /*NC=*/1);
    constexpr auto bin_block_shape = ckl::BinaryInputBlockShape::of(/*Ht=*/1, BLOCK_SIZE);

    // For non-tile-aligned W: select the partial scaler tile (idx 1) on the
    // last W-tile of the last block — accumulate_reduce / accumulate_reduce_block
    // route it that way internally.
    constexpr auto partial_scaler =
        HAS_PARTIAL_W ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // ---------- Pass 1: streaming mean for one row ----------
        ckl::accumulate_reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            cb_in, cb_scaler, cb_mean, reduce_block_shape, NUM_BLOCKS, partial_scaler);

        // ---------- Pass 2: streaming variance via (x - mean)^2 for one row ----------
        // cb_mean holds 1 tile across all blocks of pass 2 → B policy = WaitUpfrontNoPop.
        // cb_in is per-tile streamed by the reader → A policy = WaitAndPopPerTile.
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            ckl::sub<
                ckl::BroadcastDim::COL,
                ckl::BinaryInputPolicy::WaitAndPopPerTile,
                ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_in, cb_mean, cb_centered, bin_block_shape);

            ckl::square_in_place(cb_centered, bin_block_shape);

            if constexpr (COMPUTE_STD_DEV) {
                ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    cb_centered,
                    cb_scaler,
                    cb_variance,
                    reduce_block_shape,
                    b,
                    NUM_BLOCKS,
                    partial_scaler,
                    [](uint32_t dst) {
                        sqrt_tile_init();
                        sqrt_tile(dst);
                    });
            } else {
                ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    cb_centered, cb_scaler, cb_variance, reduce_block_shape, b, NUM_BLOCKS, partial_scaler);
            }
        }

        // Release this row's mean tile (held across pass 2 by WaitUpfrontNoPop).
        cb_pop_front(cb_mean, 1);

        // Drain this row's variance tile to the writer.
        ckl::copy_tiles<ckl::CopyInputPolicy::WaitAndPop>(cb_variance, cb_out, 1);
    }

    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
