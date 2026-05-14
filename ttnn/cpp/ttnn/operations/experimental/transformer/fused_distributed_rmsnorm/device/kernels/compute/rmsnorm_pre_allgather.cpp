// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
 * For rmsnorm we compute E(x**2) and return it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

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
    // Per-head mode parameters (defaults: num_heads=1, head_dim_tiles=num_tile_cols → legacy).
    // When num_heads > 1, this kernel emits `num_heads` stat tiles per row, each containing
    // sum(x**2) restricted to that head's `head_dim_tiles`-wide slice of the row.
    constexpr uint32_t num_heads = get_compile_time_arg_val(6);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(7);

    uint32_t num_tile_rows_to_process = get_arg_val<uint32_t>(0);
    constexpr uint32_t onetile = 1;

    binary_op_init_common(input_cb, input_cb, intermediate_cb);

    for (uint32_t tile_row_num = 0; tile_row_num < num_tile_rows_to_process; tile_row_num++) {
        // Per-head loop. With num_heads=1 + head_dim_tiles=num_tile_cols this collapses to
        // one iteration over the full row (identical to the legacy path).
        for (uint32_t head_idx = 0; head_idx < num_heads; head_idx++) {
            /*
             * x**2 for this head's head_dim_tiles-wide slice
             */
            reconfig_data_format(input_cb, input_cb);
            pack_reconfig_data_format(intermediate_cb);

            // Disable L1 accumulation when starting a new head's accumulation
            PACK((llk_pack_reconfig_l1_acc(0)));

            mul_tiles_init(input_cb, input_cb);
            cb_reserve_back(intermediate_cb, onetile);
            for (uint32_t col_tile = 0; col_tile < head_dim_tiles; col_tile += block_size) {
                cb_wait_front(input_cb, block_size);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size && col_tile + i < head_dim_tiles; i++) {
                    mul_tiles(input_cb, input_cb, i, i, i);
                }
                tile_regs_commit();

                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < head_dim_tiles; i++) {
                    // Pack tiles onto each other in the intermediate_cb (accumulate via L1)
                    pack_tile<true>(i /*index into DST*/, intermediate_cb, 0 /*index into intermediate CB*/);

                    if (col_tile == 0 && i == 0) {
                        // After packing the first tile in this head, enable L1 accumulation
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
                tile_regs_release();

                cb_pop_front(input_cb, block_size);
            }
            cb_push_back(intermediate_cb, onetile);

            // Disable L1 accumulation before the reduce that consumes this head's accumulator
            PACK((llk_pack_reconfig_l1_acc(0)));

            /*
             * sum(x**2) for this head — pushes one stat tile into output_cb.
             */
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, intermediate_cb, reduce_scalar_cb, output_cb>(
                compute_kernel_lib::ReduceInputBlockShape::single());          
        }
    }
    cb_pop_front(reduce_scalar_cb, onetile);
}
