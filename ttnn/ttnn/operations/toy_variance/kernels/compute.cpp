// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_variance.
//
// Computes per-row population variance via the proper two-pass algorithm:
//   variance = E[(x - E[x])^2]
//
// With scaler = 1/N built into the SUM reduce, the helpers produce:
//   Pass 1: cb_mean     = mean(x)              — accumulating reduce over blocks
//   Pass 2: cb_variance = mean((x - mean)^2)   — per-block: sub<COL> →
//                                                 square_in_place → accumulating reduce
//
// Both passes use the standard accumulate pattern: one reduce<> call per
// block with Accumulate::at(cb_acc, b), which reloads the running accumulator
// from cb_acc for b > 0. The partial scaler (and, for std-dev, the sqrt
// finalizer) are routed to the LAST block only.
//
// COMPUTE_STD_DEV: when set, sqrt is applied as the post_reduce_op on the
// pass-2 last-block reduce, so sqrt runs in DST after the final accumulation,
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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

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
    constexpr auto bin_block_shape = ckl::IterationShape::of(Ht, BLOCK_SIZE);

    // For non-tile-aligned W: select the partial scaler tile (idx 1) on the
    // last W-tile. Only the LAST block holds that tile, so the partial scaler
    // is passed on the last block and ::none() on every earlier one.
    constexpr auto partial_scaler = HAS_PARTIAL_W ? ckl::ReduceScaler::with_partial() : ckl::ReduceScaler::none();

    // ---------- Pass 1: streaming mean ----------
    // Scaler = 1/N (with partial-scaler-zeroed padded positions) converts SUM
    // into mean. One accumulating reduce<> per block into cb_mean.
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
        const bool is_last = (b + 1 == NUM_BLOCKS);
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_in, cb_scaler, cb_mean>(
            reduce_block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::Accumulate::at(cb_mean, b),
            ckl::NoOp{},
            is_last ? partial_scaler : ckl::ReduceScaler::none());
    }

    // ---------- Pass 2: streaming variance via (x - mean)^2 ----------
    // Per block:
    //   sub<COL>        : cb_in − cb_mean         → cb_centered
    //   square_in_place : cb_centered^2           → cb_centered (in-place)
    //   reduce<>        : mean(cb_centered)       → cb_variance (accumulating)
    //
    // cb_mean must persist across all blocks of pass 2 → B waits Upfront, never pops.
    // cb_in is per-tile streamed by the reader → A waits and pops per tile.
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
        // sub<COL>: cb_in − cb_mean → cb_centered.
        //   A (cb_in)   : per-tile wait/pop; each tile sits at the CB front, so Scalar idx.
        //   B (cb_mean) : waited upfront (Ht tiles, set by OperandKind::Col) and never popped
        //                 by the chain — popped manually after pass 2. COL broadcast.
        ckl::sub<
            ckl::input(cb_in, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::OperandKind::Scalar),
            ckl::input(
                cb_mean, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col),
            ckl::output(cb_centered)>(bin_block_shape);

        // square_in_place: cb_centered² → cb_centered (in-place, per-tile streaming).
        ckl::square<ckl::input(cb_centered), ckl::output(cb_centered)>(bin_block_shape);

        // Accumulating reduce into cb_variance. The last block carries the
        // partial scaler and — for std-dev — the sqrt finalizer, which the
        // reduce runs in DST after the final accumulation, before pack.
        const bool is_last = (b + 1 == NUM_BLOCKS);
        const auto block_scaler = is_last ? partial_scaler : ckl::ReduceScaler::none();

        if constexpr (COMPUTE_STD_DEV) {
            if (is_last) {
                ckl::
                    reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_centered, cb_scaler, cb_variance>(
                        reduce_block_shape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at(cb_variance, b),
                        [](uint32_t dst) {
                            sqrt_tile_init();
                            sqrt_tile(dst);
                        },
                        block_scaler);
                continue;
            }
        }

        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_centered, cb_scaler, cb_variance>(
            reduce_block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::Accumulate::at(cb_variance, b),
            ckl::NoOp{},
            block_scaler);
    }

    // cb_mean was held across pass 2 (Upfront wait, no pop) — release it now.
    cb_pop_front(cb_mean, Ht);

    // ---------- Drain cb_variance → cb_out ----------
    // Per-tile streaming copy with input + output format reconfig (chain owns
    // wait/pop on cb_variance and reserve/push on cb_out).
    ckl::copy<ckl::input(cb_variance), ckl::output(cb_out)>(ckl::IterationShape::tiles(Ht));

    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
