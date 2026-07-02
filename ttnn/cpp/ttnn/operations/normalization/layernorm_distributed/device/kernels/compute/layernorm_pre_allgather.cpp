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
#include "ttnn/operations/normalization/kernel_util/compute/pre_add.h"

namespace pre_add = norm::kernel_util::compute::pre_add;

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce_id = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_x2_id = tt::CBIndex::c_6;   // x**2
    constexpr uint32_t cb_res_id = tt::CBIndex::c_5;  // residual b (unused when !FUSE_PRE_ADD)
    constexpr uint32_t cb_inp_id = FUSE_PRE_ADD ? tt::CBIndex::c_3 : cb_in0_id;  // fused a + b, or just a

    if constexpr (FUSE_PRE_ADD) {
        binary_op_init_common(cb_in0_id, cb_res_id, cb_inp_id);
    } else {
        binary_op_init_common(cb_inp_id, cb_reduce_id, cb_x2_id);
    }

    DataflowBuffer cb_in0(cb_in0_id);
    DataflowBuffer cb_res(cb_res_id);
    DataflowBuffer cb_inp(cb_inp_id);
    DataflowBuffer cb_x2(cb_x2_id);
    DataflowBuffer cb_reduce(cb_reduce_id);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: cb_inp_id = cb_in0_id + cb_res_id (no-op when !FUSE_PRE_ADD)
        pre_add::one_row<FUSE_PRE_ADD>(cb_in0, cb_res, cb_inp, Wt, blk);

        /*
         * x**2
         */
        reconfig_data_format(cb_inp_id, cb_inp_id);
        pack_reconfig_data_format(cb_x2_id);
        mul_tiles_init(cb_inp_id, cb_inp_id);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_inp.wait_front(wt + blk);  // cumulative wait

            tile_regs_acquire();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(cb_inp_id, cb_inp_id, wt + wtr, wt + wtr, wtr);
            }
            tile_regs_commit();

            cb_x2.reserve_back(blk);

            tile_regs_wait();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                pack_tile(wtr, cb_x2_id, wt + wtr);
            }
            tile_regs_release();

            cb_x2.push_back(blk);
        }
        /*
         * sum(x**2)
         */

        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        // Bulk mode for optimal performance
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            cb_x2_id,
            cb_reduce_id,
            cb_out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        /*
         * sum(x)
         */
        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        // Bulk mode for optimal performance
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            cb_inp_id,
            cb_reduce_id,
            cb_out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
    }
    cb_reduce.pop_front(1);
}
