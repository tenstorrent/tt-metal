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

    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_reduce_id = tt::CBIndex::c_1;

    constexpr uint32_t dfb_out = tt::CBIndex::c_14;

    constexpr uint32_t dfb_x2_id = tt::CBIndex::c_6;   // x**2
    constexpr uint32_t dfb_res_id = tt::CBIndex::c_5;  // residual b (unused when !FUSE_PRE_ADD)
    constexpr uint32_t dfb_inp_id = FUSE_PRE_ADD ? tt::CBIndex::c_3 : dfb_in0_id;  // fused a + b, or just a

    if constexpr (FUSE_PRE_ADD) {
        binary_op_init_common(dfb_in0_id, dfb_res_id, dfb_inp_id);
    } else {
        binary_op_init_common(dfb_inp_id, dfb_reduce_id, dfb_x2_id);
    }

    DataflowBuffer dfb_in0(dfb_in0_id);
    DataflowBuffer dfb_res(dfb_res_id);
    DataflowBuffer dfb_inp(dfb_inp_id);
    DataflowBuffer dfb_x2(dfb_x2_id);
    DataflowBuffer dfb_reduce(dfb_reduce_id);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: cb_inp_id = cb_in0_id + cb_res_id (no-op when !FUSE_PRE_ADD)
        pre_add::one_row<FUSE_PRE_ADD>(dfb_in0, dfb_res, dfb_inp, Wt, blk);

        /*
         * x**2
         */
        reconfig_data_format(dfb_inp_id, dfb_inp_id);
        pack_reconfig_data_format(dfb_x2_id);
        mul_tiles_init(dfb_inp_id, dfb_inp_id);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            dfb_inp.wait_front(wt + blk);  // cumulative wait

            tile_regs_acquire();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(dfb_inp_id, dfb_inp_id, wt + wtr, wt + wtr, wtr);
            }
            tile_regs_commit();

            dfb_x2.reserve_back(blk);

            tile_regs_wait();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                pack_tile(wtr, dfb_x2_id, wt + wtr);
            }
            tile_regs_release();

            dfb_x2.push_back(blk);
        }
        /*
         * sum(x**2)
         */

        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        // Bulk mode for optimal performance
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            dfb_x2_id,
            dfb_reduce_id,
            dfb_out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        /*
         * sum(x)
         */
        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        // Bulk mode for optimal performance
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            dfb_inp_id,
            dfb_reduce_id,
            dfb_out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
    }
    dfb_reduce.pop_front(1);
}
