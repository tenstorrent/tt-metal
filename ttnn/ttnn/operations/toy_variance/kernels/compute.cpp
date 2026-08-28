// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_variance (interleaved input, single core).
//
// Computes per-row population variance via the proper two-pass algorithm:
//   variance = E[(x - E[x])^2]
//
// With an AVG reduce over REDUCE_N elements (scaler = 1/N, owned by this kernel), the helpers
// produce:
//   Pass 1: dfb::mean     = mean(x)              -- accumulating reduce over blocks
//   Pass 2: dfb::variance = mean((x - mean)^2)   -- per-block: sub<COL> -> square_in_place ->
//                                                  accumulating reduce
//
// Both passes use the standard accumulate pattern: one reduce<> call per block with
// Accumulate::at(dfb_acc, b), which reloads the running accumulator for b > 0. The partial scaler
// (and, for std-dev, the sqrt finalizer) are routed to the LAST block only.
//
// The scaler tiles live in dfb::scaler but are built and reused by reduce<> itself via
// ReduceScaler::compute_managed(); no reader writes them and nothing pops them here.
//
// compute_std_dev: when set, sqrt is applied as the post_reduce_op on the pass-2 last-block reduce,
// so sqrt runs in DST after the final accumulation, before pack -- no extra pass over the data, and
// intermediate accumulator values stay in variance space.
//
// All work goes through the kernel-lib helpers (no raw tile_regs / copy_tile / pack_tile loops in
// this kernel) -- reducing the surface for buffer-sync, DST, and reconfig bugs.
//
// Metal 2.0: the helper library is Gen1 and takes raw buffer ids, but the `dfb::` names are passed
// to it directly -- DFBBindingToken's conversion to uint32_t is constexpr, so a binding token is a
// valid non-type template argument. There is no second name for any buffer in this kernel.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t BLOCK_SIZE = get_arg(args::block_size);
    constexpr uint32_t NUM_BLOCKS = get_arg(args::num_blocks);
    constexpr bool COMPUTE_STD_DEV = get_arg(args::compute_std_dev) != 0;
    constexpr bool HAS_PARTIAL_W = get_arg(args::has_partial_w) != 0;
    constexpr uint32_t PARTIAL_W = get_arg(args::partial_w);  // valid positions in last W-tile
    constexpr uint32_t REDUCE_N = get_arg(args::reduce_n);    // real W -> AVG scaler is 1/REDUCE_N

    compute_kernel_hw_startup(dfb::in_tiles, dfb::scaler, dfb::out_tiles);

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(Ht, BLOCK_SIZE, /*NC=*/1);
    constexpr auto bin_block_shape = ckl::IterationShape::of(Ht, BLOCK_SIZE);

    // For non-tile-aligned W: the last W-tile needs a scaler that zeros the padded positions. Only
    // the LAST block holds that tile, so the partial variant is passed on the last block and the
    // plain full-tile variant on every earlier one.
    constexpr auto partial_scaler = ckl::ReduceScaler::compute_managed(HAS_PARTIAL_W ? PARTIAL_W : 0);
    constexpr auto full_scaler = ckl::ReduceScaler::compute_managed(0);

    // ---------- Pass 1: streaming mean ----------
    // AVG over REDUCE_N (with partial-scaler-zeroed padded positions) yields the row mean. One
    // accumulating reduce<> per block into dfb::mean.
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
        const bool is_last = (b + 1 == NUM_BLOCKS);
        ckl::reduce<
            ckernel::PoolType::AVG,
            ckernel::ReduceDim::REDUCE_ROW,
            dfb::in_tiles,
            dfb::scaler,
            dfb::mean,
            ckl::ReduceInputPolicy::WaitAndPopPerTile,
            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ckl::ReduceAlgorithm::Auto,
            ckl::ReduceWithinTile::Collapse,
            REDUCE_N>(
            reduce_block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::Accumulate::at(dfb::mean, b),
            ckl::NoOp{},
            is_last ? partial_scaler : full_scaler);
    }

    // ---------- Pass 2: streaming variance via (x - mean)^2 ----------
    // Per block:
    //   sub<COL>        : in_tiles - mean          -> centered_sq
    //   square_in_place : centered_sq^2            -> centered_sq (in-place)
    //   reduce<>        : mean(centered_sq)        -> variance (accumulating)
    //
    // dfb::mean must persist across all blocks of pass 2 -> B waits Upfront, never pops.
    // dfb::in_tiles is per-tile streamed by the reader -> A waits and pops per tile.
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
        ckl::sub<
            ckl::input(dfb::in_tiles, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::OperandKind::Scalar),
            ckl::input(
                dfb::mean,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::None,
                ckl::OperandKind::Col),
            ckl::output(dfb::centered_sq)>(bin_block_shape);

        ckl::square<ckl::input(dfb::centered_sq), ckl::output(dfb::centered_sq)>(bin_block_shape);

        const bool is_last = (b + 1 == NUM_BLOCKS);
        const auto block_scaler = is_last ? partial_scaler : full_scaler;

        if constexpr (COMPUTE_STD_DEV) {
            if (is_last) {
                ckl::reduce<
                    ckernel::PoolType::AVG,
                    ckernel::ReduceDim::REDUCE_ROW,
                    dfb::centered_sq,
                    dfb::scaler,
                    dfb::variance,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    ckl::ReduceAlgorithm::Auto,
                    ckl::ReduceWithinTile::Collapse,
                    REDUCE_N>(
                    reduce_block_shape,
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(dfb::variance, b),
                    [](uint32_t dst) {
                        sqrt_tile_init();
                        sqrt_tile(dst);
                    },
                    block_scaler);
                continue;
            }
        }

        ckl::reduce<
            ckernel::PoolType::AVG,
            ckernel::ReduceDim::REDUCE_ROW,
            dfb::centered_sq,
            dfb::scaler,
            dfb::variance,
            ckl::ReduceInputPolicy::WaitAndPopPerTile,
            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ckl::ReduceAlgorithm::Auto,
            ckl::ReduceWithinTile::Collapse,
            REDUCE_N>(
            reduce_block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::Accumulate::at(dfb::variance, b),
            ckl::NoOp{},
            block_scaler);
    }

    DataflowBuffer dfb_mean(dfb::mean);

    // dfb::mean was held across pass 2 (Upfront wait, no pop) -- release it now.
    dfb_mean.pop_front(Ht);

    // ---------- Drain variance -> out_tiles ----------
    // Per-tile streaming copy with input + output format reconfig (chain owns wait/pop on
    // dfb::variance and reserve/push on dfb::out_tiles).
    ckl::copy<ckl::input(dfb::variance), ckl::output(dfb::out_tiles)>(ckl::IterationShape::tiles(Ht));
}
