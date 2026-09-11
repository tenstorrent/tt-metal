// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_variance (interleaved input, single core).
//
// Computes per-row population variance via the proper two-pass algorithm:
//   variance = E[(x - E[x])^2]
//
// With scaler = 1/N built into the SUM reduce, the helpers produce:
//   Pass 1: dfb::mean     = mean(x)              -- accumulating reduce over blocks
//   Pass 2: dfb::variance = mean((x - mean)^2)   -- per-block: sub<COL> -> square_in_place ->
//                                                  accumulating reduce
//
// Each pass uses a host-planned seed/repeat/final sequence. The auxiliary recipe
// carries the final partial tile, and callbacks run only after final accumulation.
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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

namespace ckl = compute_kernel_lib;

template <uint32_t Input, uint32_t Output, typename PostOp>
ALWI void reduce_block(uint32_t block, PostOp post_op) {
    constexpr uint32_t count = get_compile_time_arg_val(0);
    using First = ttnn::kernel_lib::
        BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, 0>, Input, dfb::scaler, Output, dfb::reduce_accumulator>;
    using Middle = ttnn::kernel_lib::BoundReduceCallArgs<
        ttnn::kernel_lib::ReduceCallAtT<1, (count > 1 ? 1 : 0)>,
        Input,
        dfb::scaler,
        Output,
        dfb::reduce_accumulator>;
    using Last = ttnn::kernel_lib::BoundReduceCallArgs<
        ttnn::kernel_lib::ReduceCallAtT<1, count - 1>,
        Input,
        dfb::scaler,
        Output,
        dfb::reduce_accumulator>;
    if (block + 1 == get_arg(args::num_blocks)) {
        ckl::reduce<Last>(post_op);
    } else if (block == 0) {
        ckl::reduce<First>();
    } else {
        ckl::reduce<Middle>();
    }
}

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t BLOCK_SIZE = get_arg(args::block_size);
    constexpr uint32_t NUM_BLOCKS = get_arg(args::num_blocks);
    constexpr bool COMPUTE_STD_DEV = get_arg(args::compute_std_dev) != 0;

    compute_kernel_hw_startup(dfb::in_tiles, dfb::scaler, dfb::out_tiles);

    constexpr auto bin_block_shape = ckl::IterationShape::of(Ht, BLOCK_SIZE);

    // ---------- Pass 1: streaming mean ----------
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
        reduce_block<dfb::in_tiles, dfb::mean>(b, ckl::NoOp{});
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

        reduce_block<dfb::centered_sq, dfb::variance>(b, [](uint32_t dst) {
            if constexpr (COMPUTE_STD_DEV) {
                sqrt_tile_init();
                sqrt_tile(dst);
            }
        });
    }

    DataflowBuffer dfb_mean(dfb::mean);
    DataflowBuffer dfb_scaler(dfb::scaler);

    // dfb::mean was held across pass 2 (Upfront wait, no pop) -- release it now.
    dfb_mean.pop_front(Ht);

    // ---------- Drain variance -> out_tiles ----------
    // Per-tile streaming copy with input + output format reconfig (chain owns wait/pop on
    // dfb::variance and reserve/push on dfb::out_tiles).
    ckl::copy<ckl::input(dfb::variance), ckl::output(dfb::out_tiles)>(ckl::IterationShape::tiles(Ht));

    dfb_scaler.pop_front(get_arg(args::auxiliary_tiles));
}
