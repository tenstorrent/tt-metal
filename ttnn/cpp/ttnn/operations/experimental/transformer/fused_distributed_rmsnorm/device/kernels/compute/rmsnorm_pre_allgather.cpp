// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
 * For rmsnorm we compute E(x**2) and return it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/debug/dprint_pages.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);
    // Note: get_compile_time_arg_val(6) is use_float32_reduction - unused after library migration
    // Library auto-detects FP32 from ENABLE_FP32_DEST_ACC define

    uint32_t num_tile_rows_to_process = get_arg_val<uint32_t>(0);
    constexpr uint32_t onetile = 1;

    binary_op_init_common(input_cb, input_cb, intermediate_cb);

    for (uint32_t tile_row_num = 0; tile_row_num < num_tile_rows_to_process; tile_row_num++) {
        /*
         * x**2
         */
        reconfig_data_format(input_cb, input_cb);
        pack_reconfig_data_format(intermediate_cb);

        // Disable L1 accumulation when starting a new row
        PACK((llk_pack_reconfig_l1_acc(0)));

        mul_tiles_init(input_cb, input_cb);
        cb_reserve_back(intermediate_cb, onetile);
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_wait_front(input_cb, block_size);

            tile_regs_acquire();
            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                mul_tiles(input_cb, input_cb, i, i, i);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                // Pack tiles onto eachother in the intermediate_cb
                pack_tile<true>(i /*index into DST*/, intermediate_cb, 0 /*index into intermediate CB*/);

                if (col_tile == 0 && i == 0) {
                    // After packing the first tile in this row, enable L1 accumulation
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            tile_regs_release();

            cb_pop_front(input_cb, block_size);
        }
        cb_push_back(intermediate_cb, onetile);

        // Disable L1 accumulation
        PACK((llk_pack_reconfig_l1_acc(0)));

        /*
         * sum(x**2)
         */
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            intermediate_cb, reduce_scalar_cb, output_cb, compute_kernel_lib::TileShape::single());
    }
    cb_pop_front(reduce_scalar_cb, onetile);
}
