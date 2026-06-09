// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_variance.
//
// Computes per-row population variance via the proper two-pass algorithm:
//   variance = E[(x - E[x])^2]
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
//
// All work goes through the kernel-lib helpers (no raw tile_regs / copy_tile /
// pack_tile loops in this kernel) — reducing the surface for CB-sync, DST,
// and reconfig bugs.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
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

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(Ht, BLOCK_SIZE, /*NC=*/1);
    constexpr auto bin_block_shape = ckl::EltwiseShape::of(Ht, BLOCK_SIZE);

    // For non-tile-aligned W: select the partial scaler tile (idx 1) on the
    // last W-tile of the last block — accumulate_reduce / accumulate_reduce_block
    // route it that way internally.
    constexpr auto partial_scaler =
        HAS_PARTIAL_W ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    // ---------- Pass 1: streaming mean ----------
    // Scaler = 1/N (with partial-scaler-zeroed padded positions) converts SUM
    // into mean. accumulate_reduce owns the block loop.
    ckl::accumulate_reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        cb_in, cb_scaler, cb_mean, reduce_block_shape, NUM_BLOCKS, partial_scaler);

    // ---------- Pass 2: streaming variance via (x - mean)^2 ----------
    // Per block:
    //   sub<COL>           : cb_in − cb_mean         → cb_centered
    //   square_in_place    : cb_centered^2          → cb_centered (in-place)
    //   accumulate_reduce_block : mean(cb_centered) → cb_variance
    //
    // cb_mean must persist across all blocks of pass 2 → B policy = WaitUpfrontNoPop.
    // cb_in is per-tile streamed by the reader → A policy = WaitAndPopPerTile.
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
        // sub<COL>: cb_in − cb_mean → cb_centered.
        //   A (cb_in)   : WaitAndPopPerTile → Streaming, per-tile front-relative (Scalar idx).
        //   B (cb_mean) : WaitUpfrontNoPop  → HeldBulk (no pop; popped manually after pass 2),
        //                 COL broadcast → OperandKind::Col.
        ckl::sub<
            cb_in,
            cb_mean,
            cb_centered,
            ckl::BroadcastDim::Col,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::HeldBulk,
            compute_kernel_lib::OutputLifecycle::Streaming,
            ckl::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            ckl::OperandKind::Scalar,
            ckl::OperandKind::Col>(bin_block_shape);  // B index (COL broadcast)

        // square_in_place: cb_centered² → cb_centered (in-place, per-tile streaming).
        ckl::square<cb_centered, cb_centered>(bin_block_shape);

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

    // cb_mean was held across pass 2 with WaitUpfrontNoPop — release it now.
    cb_pop_front(cb_mean, Ht);

    // ---------- Drain cb_variance → cb_out ----------
    // Per-tile streaming copy with input + output format reconfig (chain owns
    // wait/pop on cb_variance and reserve/push on cb_out).
    ckl::copy<cb_variance, cb_out>(Ht);

    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
