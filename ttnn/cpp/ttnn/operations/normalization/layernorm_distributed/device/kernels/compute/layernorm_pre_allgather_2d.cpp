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
#include "ttnn/operations/normalization/kernel_util/compute/pre_add.h"

namespace pre_add = norm::kernel_util::compute::pre_add;

void kernel_main() {
    constexpr uint32_t NCHt = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(3);
    bool is_merge_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t dfb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_reduce_id = tt::CBIndex::c_1;

    constexpr uint32_t dfb_out = tt::CBIndex::c_16;

    constexpr uint32_t dfb_x2_id = tt::CBIndex::c_6;  // x**2
    constexpr uint32_t dfb_zero_id = tt::CBIndex::c_13;
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
    DataflowBuffer dfb_zero(dfb_zero_id);

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
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            dfb_x2_id,
            dfb_reduce_id,
            dfb_out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        dfb_inp.pop_front(Wt);
        dfb_reduce.pop_front(1);
    }

    // if merge core, we need to do a final sum on the tile in cb_x2_id and then write the result to cb_out_final_id
    if (is_merge_core) {
        constexpr uint32_t dfb_x2_merge_id = tt::CBIndex::c_15;
        constexpr uint32_t dfb_out_final_id = tt::CBIndex::c_14;
        DataflowBuffer dfb_x2_merge(dfb_x2_merge_id);
        DataflowBuffer dfb_out_final(dfb_out_final_id);
        constexpr int dst0 = 0;

        // Wait for all num_cores_y tiles
        dfb_x2_merge.wait_front(num_cores_y);
        dfb_zero.wait_front(1);

        // Initialize accumulation
        binary_op_init_common(dfb_x2_merge_id, dfb_zero_id, dfb_out_final_id);
        reconfig_data_format(dfb_x2_merge_id, dfb_zero_id);
        pack_reconfig_data_format(dfb_out_final_id);
        add_tiles_init(dfb_x2_merge_id, dfb_zero_id, true);

        tile_regs_acquire();
        // Add all 8 tiles together
        for (uint32_t i = 0; i < num_cores_y; i++) {
            add_tiles(dfb_x2_merge_id, dfb_zero_id, i, 0, dst0);
        }
        tile_regs_commit();

        dfb_x2_merge.pop_front(num_cores_y);

        dfb_out_final.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(dst0, dfb_out_final_id);
        tile_regs_release();

        dfb_out_final.push_back(onetile);
    }
}
