// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "api/debug/dprint.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t NCHt = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(3);
    // Note: get_compile_time_arg_val(4) is FLOAT32_REDUCTION - unused after library migration
    // Library auto-detects FP32 from ENABLE_FP32_DEST_ACC define
    bool is_merge_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce_idx = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t cb_x2_idx = tt::CBIndex::c_6;  // x**2
    constexpr uint32_t cb_zero_idx = tt::CBIndex::c_13;

    binary_op_init_common(cb_inp_idx, cb_reduce_idx, cb_x2_idx);

    CircularBuffer cb_inp(cb_inp_idx);
    CircularBuffer cb_reduce(cb_reduce_idx);
    CircularBuffer cb_x2(cb_x2_idx);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * x**2
         */
        reconfig_data_format(cb_inp_idx, cb_inp_idx);
        pack_reconfig_data_format(cb_x2_idx);
        mul_tiles_init(cb_inp_idx, cb_inp_idx);

        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_inp.wait_front(wt + blk);  // cumulative wait

            cb_x2.reserve_back(blk);
            ACQ();

            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(cb_inp_idx, cb_inp_idx, wt + wtr, wt + wtr, wtr);
                pack_tile(wtr, cb_x2_idx, wt + wtr);
            }
            REL();

            cb_x2.push_back(blk);
        }

        /*
         * sum(x**2)
         */
        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            cb_x2_idx,
            cb_reduce_idx,
            cb_out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        cb_inp.pop_front(Wt);
        cb_reduce.pop_front(1);
    }

    // if merge core, we need to do a final sum on the tile in cb_x2_idx and then write the result to cb_out_final_idx
    if (is_merge_core) {
        constexpr uint32_t cb_x2_merge_idx = tt::CBIndex::c_15;
        constexpr uint32_t cb_out_final_idx = tt::CBIndex::c_14;
        constexpr int dst0 = 0;

        CircularBuffer cb_x2_merge(cb_x2_merge_idx);
        CircularBuffer cb_zero(cb_zero_idx);
        CircularBuffer cb_out_final(cb_out_final_idx);

        // Wait for all num_cores_y tiles
        cb_x2_merge.wait_front(num_cores_y);
        cb_zero.wait_front(1);

        // Reserve output space
        cb_out_final.reserve_back(onetile);

        // Initialize accumulation
        binary_op_init_common(cb_x2_merge_idx, cb_zero_idx, cb_out_final_idx);
        reconfig_data_format(cb_x2_merge_idx, cb_zero_idx);
        pack_reconfig_data_format(cb_out_final_idx);
        add_tiles_init(cb_x2_merge_idx, cb_zero_idx, true);

        // Acquire registers
        ACQ();

        // Add all 8 tiles together
        for (uint32_t i = 0; i < num_cores_y; i++) {
            add_tiles(cb_x2_merge_idx, cb_zero_idx, i, 0, dst0);
        }

        // Pack result
        pack_tile(dst0, cb_out_final_idx);
        REL();

        // Push output and pop input
        cb_out_final.push_back(onetile);
        cb_x2_merge.pop_front(num_cores_y);
    }
}
