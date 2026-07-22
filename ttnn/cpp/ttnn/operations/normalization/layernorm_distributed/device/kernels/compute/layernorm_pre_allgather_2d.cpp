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

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce_id = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t cb_x2_id = tt::CBIndex::c_6;  // x**2
    constexpr uint32_t cb_zero_id = tt::CBIndex::c_13;
    constexpr uint32_t cb_res_id = tt::CBIndex::c_5;  // residual b (unused when !FUSE_PRE_ADD)
    constexpr uint32_t cb_inp_id = FUSE_PRE_ADD ? tt::CBIndex::c_3 : cb_in0_id;  // fused a + b, or just a

    if constexpr (FUSE_PRE_ADD) {
        compute_kernel_hw_startup(cb_in0_id, cb_res_id, cb_inp_id);
    } else {
        compute_kernel_hw_startup(cb_inp_id, cb_reduce_id, cb_x2_id);
    }

    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_res(cb_res_id);
    CircularBuffer cb_inp(cb_inp_id);
    CircularBuffer cb_x2(cb_x2_id);
    CircularBuffer cb_reduce(cb_reduce_id);
    CircularBuffer cb_zero(cb_zero_id);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: cb_inp_id = cb_in0_id + cb_res_id (no-op when !FUSE_PRE_ADD)
        pre_add::one_row<FUSE_PRE_ADD>(cb_in0, cb_res, cb_inp, Wt, blk);

        /*
         * x**2
         */
        reconfig_data_format(cb_inp_id, cb_inp_id);
        pack_reconfig_data_format(cb_x2_id);
        mul_init(cb_inp_id, cb_inp_id);

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
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            cb_x2_id,
            cb_reduce_id,
            cb_out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        cb_inp.pop_front(Wt);
        cb_reduce.pop_front(1);
    }

    // if merge core, we need to do a final sum on the tile in cb_x2_id and then write the result to cb_out_final_id
    if (is_merge_core) {
        constexpr uint32_t cb_x2_merge_id = tt::CBIndex::c_15;
        constexpr uint32_t cb_out_final_id = tt::CBIndex::c_14;
        CircularBuffer cb_x2_merge(cb_x2_merge_id);
        CircularBuffer cb_out_final(cb_out_final_id);
        constexpr int dst0 = 0;

        // Wait for all num_cores_y tiles
        cb_x2_merge.wait_front(num_cores_y);
        cb_zero.wait_front(1);

        // Initialize accumulation
        compute_kernel_hw_startup(cb_x2_merge_id, cb_zero_id, cb_out_final_id);
        reconfig_data_format(cb_x2_merge_id, cb_zero_id);
        pack_reconfig_data_format(cb_out_final_id);
        add_init(cb_x2_merge_id, cb_zero_id, true);

        tile_regs_acquire();
        // Add all 8 tiles together
        for (uint32_t i = 0; i < num_cores_y; i++) {
            add_tiles(cb_x2_merge_id, cb_zero_id, i, 0, dst0);
        }
        tile_regs_commit();

        cb_x2_merge.pop_front(num_cores_y);

        cb_out_final.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(dst0, cb_out_final_id);
        tile_regs_release();

        cb_out_final.push_back(onetile);
    }
}
