// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes larnorm statistics.
 * For layernorm it computes E(x**2) and E(x) and returns them as a two tile wide output tensor containing E(x**2) and
 * E(x) in the left most columns per tile. For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    // Note: get_compile_time_arg_val(2) is FLOAT32_REDUCTION - unused after library migration
    // Library auto-detects FP32 from ENABLE_FP32_DEST_ACC define

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;  // x**2

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * x**2
         */
        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);
        mul_tiles_init(cb_inp, cb_inp);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_inp, wt + blk);  // cumulative wait
            cb_reserve_back(cb_x2, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(cb_inp, cb_inp, wt + wtr, wt + wtr, wtr);
                pack_tile(wtr, cb_x2, wt + wtr);
            }
            REL();
            cb_push_back(cb_x2, blk);
        }
        /*
         * sum(x**2)
         */

        // STREAMING_BATCHED: All Wt tiles already in CB (see cumulative wait above)
        // Batched mode for optimal performance
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerBatch>(
                cb_x2, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        /*
         * sum(x)
         */
        // STREAMING_BATCHED: All Wt tiles already in CB (see cumulative wait above)
        // Batched mode for optimal performance
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerBatch>(
                cb_inp, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
    }
    cb_pop_front(cb_reduce, 1);
}
