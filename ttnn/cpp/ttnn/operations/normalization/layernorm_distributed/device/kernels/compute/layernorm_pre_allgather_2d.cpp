// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "debug/dprint.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t NCHt = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(3);
    bool is_merge_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;  // x**2
    constexpr uint32_t cb_zero = tt::CBIndex::c_13;

    cb_wait_front(cb_reduce, 1);  // comes from the reader

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int dst0 = 0;

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

        reconfig_data_format(cb_x2, cb_reduce);
        pack_reconfig_data_format(cb_out);
        reduce_init(cb_x2, cb_reduce, cb_out);

        cb_wait_front(cb_x2, Wt);

        cb_reserve_back(cb_out, onetile);
        ACQ();

        for (uint32_t wtr = 0; wtr < Wt; wtr++) {
            reduce_tile(cb_x2, cb_reduce, wtr, 0, dst0);
        }
        pack_tile(dst0, cb_out, 0);
        REL();

        cb_push_back(cb_out, onetile);

        cb_pop_front(cb_x2, Wt);

        reduce_uninit();

        cb_pop_front(cb_inp, Wt);
        cb_pop_front(cb_reduce, 1);
    }

    // if merge core, we need to do a final sum on the tile in cb_x2 and then write the result to cb_out_final
    if (is_merge_core) {
        constexpr uint32_t cb_x2_merge = tt::CBIndex::c_15;
        constexpr uint32_t cb_out_final = tt::CBIndex::c_14;
        constexpr int dst0 = 0;

        // Wait for all num_cores_y tiles
        cb_wait_front(cb_x2_merge, num_cores_y);
        cb_wait_front(cb_zero, 1);

        // Reserve output space
        cb_reserve_back(cb_out_final, onetile);

        // Initialize accumulation
        binary_op_init_common(cb_x2_merge, cb_zero, cb_out_final);
        reconfig_data_format(cb_x2_merge, cb_zero);
        pack_reconfig_data_format(cb_out_final);
        add_tiles_init(cb_x2_merge, cb_zero, true);

        // Acquire registers
        ACQ();

        // Add all 8 tiles together
        for (uint32_t i = 0; i < num_cores_y; i++) {
            add_tiles(cb_x2_merge, cb_zero, i, 0, dst0);
        }

        // Pack result
        pack_tile(dst0, cb_out_final);
        REL();

        // Push output and pop input
        cb_push_back(cb_out_final, onetile);
        cb_pop_front(cb_x2_merge, num_cores_y);
    }
}
}  // namespace NAMESPACE
