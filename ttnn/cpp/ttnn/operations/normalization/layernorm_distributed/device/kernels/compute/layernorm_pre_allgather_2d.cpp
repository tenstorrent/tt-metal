// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
For rmsnorm it computes E(x**2) and returns it as a one tile wide output
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

    constexpr uint32_t NCHt = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(3);
    bool is_merge_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;  // x**2
    constexpr uint32_t cb_zero = tt::CBIndex::c_13;

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * x**2 — helper absorbs the chunked-cumulative outer loop.
         * shape.cols = Wt, wait_step = blk (per-chunk granularity).
         * One init+reconfig per call.
         */
        square<BinaryInputPolicy::CumulativeWaitNoPop, BinaryOutputPolicy::Bulk>(
            cb_inp, cb_x2, BinaryInputBlockShape::of(1, Wt), NoOp{}, BinaryInputExtras{.wait_step = blk});

        /*
         * sum(x**2)
         */
        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_x2, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        cb_pop_front(cb_inp, Wt);
        cb_pop_front(cb_reduce, 1);
    }

    // Merge core: sum num_cores_y partial E(x²) tiles into a single tile via hardware
    // acc_to_dest. A streams cb_x2_merge[0..num_cores_y); B stays at cb_zero[0] (additive
    // zero-partner). Helper emits one output tile and manages reserve/push; A policy
    // WaitUpfrontPopAtEnd covers the upfront wait and bulk pop.
    if (is_merge_core) {
        constexpr uint32_t cb_x2_merge = tt::CBIndex::c_15;
        constexpr uint32_t cb_out_final = tt::CBIndex::c_14;

        binary_op_init_common(cb_x2_merge, cb_zero, cb_out_final);
        cb_wait_front(cb_zero, 1);

        add<BroadcastDim::NONE,
            BinaryInputPolicy::WaitUpfrontPopAtEnd,
            BinaryInputPolicy::NoWaitNoPop,
            BinaryOutputPolicy::AccumulateInDest>(
            cb_x2_merge, cb_zero, cb_out_final, BinaryInputBlockShape::of(1, num_cores_y));
    }
}
