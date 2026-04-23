// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes larnorm statistics.
 * For layernorm it computes E(x**2) and E(x) and returns them as a two tile wide output tensor containing E(x**2) and
 * E(x) in the left most columns per tile. For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;  // x**2

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * x**2 — helper absorbs the hand-rolled chunked-cumulative outer loop.
         * shape.cols = Wt (full block), wait_step = blk (per-chunk granularity).
         * Helper's internal chunk loop issues cb_wait_front(cb_inp, group_end)
         * with group_end growing by blk per chunk, matching the pre-migration
         * `cb_wait_front(cb_inp, wt + blk)` pattern. One init+reconfig per call.
         * No pop — the later reductions consume cb_inp's tiles.
         */
        square<BinaryInputPolicy::CumulativeWaitNoPop, BinaryOutputPolicy::Bulk>(
            cb_inp, cb_x2, BinaryInputBlockShape::of(1, Wt), NoOp{}, BinaryInputExtras{.wait_step = blk});
        /*
         * sum(x**2)
         */

        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        // Bulk mode for optimal performance
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_x2, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        /*
         * sum(x)
         */
        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        // Bulk mode for optimal performance
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_inp, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
    }
    cb_pop_front(cb_reduce, 1);
}
