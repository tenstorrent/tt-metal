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
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);

    uint32_t num_tile_rows_to_process = get_arg_val<uint32_t>(0);
    constexpr uint32_t onetile = 1;

    CircularBuffer cb_input(input_cb);
    CircularBuffer cb_reduce_scalar(reduce_scalar_cb);
    CircularBuffer cb_intermediate(intermediate_cb);

    compute_kernel_hw_startup(input_cb, input_cb, intermediate_cb);

    constexpr uint32_t call_count = get_compile_time_arg_val(6);
    using First = ttnn::kernel_lib::ReduceCallAtT<7, 0>;
    using Middle = ttnn::kernel_lib::ReduceCallAtT<7, (call_count > 1 ? 1 : 0)>;
    using Last = ttnn::kernel_lib::ReduceCallAtT<7, call_count - 1>;

    for (uint32_t tile_row_num = 0; tile_row_num < num_tile_rows_to_process; ++tile_row_num) {
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const uint32_t tiles = (num_tile_cols - col_tile < block_size) ? num_tile_cols - col_tile : block_size;
            reconfig_data_format(input_cb, input_cb);
            pack_reconfig_data_format(intermediate_cb);
            mul_init(input_cb, input_cb);
            cb_input.wait_front(block_size);
            cb_intermediate.reserve_back(block_size);
            tile_regs_acquire();
            for (uint32_t i = 0; i < tiles; ++i) {
                mul_tiles(input_cb, input_cb, i, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < tiles; ++i) {
                pack_tile(i, intermediate_cb);
            }
            tile_regs_release();
            cb_input.pop_front(block_size);
            cb_intermediate.push_back(block_size);
            cb_intermediate.wait_front(block_size);
            if (col_tile + block_size >= num_tile_cols) {
                compute_kernel_lib::reduce<Last>();
            } else if (col_tile == 0) {
                compute_kernel_lib::reduce<First>();
            } else {
                compute_kernel_lib::reduce<Middle>();
            }
            cb_intermediate.pop_front(block_size);
        }
    }
    cb_reduce_scalar.pop_front(get_named_compile_time_arg_val("reduce_auxiliary_tiles"));
}
