// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

constexpr uint32_t compute_input_cb_id_0 = get_compile_time_arg_val(0);
constexpr uint32_t compute_input_cb_id_1 = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(3);
constexpr uint32_t input_granularity = get_compile_time_arg_val(4);
constexpr uint32_t reduction_dim = get_compile_time_arg_val(5);
constexpr uint32_t reduction_dim_size = get_compile_time_arg_val(6);
constexpr uint32_t inner_num_tiles = get_compile_time_arg_val(7);
constexpr uint32_t reduction_num_tiles = get_compile_time_arg_val(8);

constexpr uint32_t initial_ct_idx = 9;

void kernel_main() {
    uint32_t arg_idx = 0;

    uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    // TensorAccessor
    constexpr auto tensor_args = TensorAccessorArgs<initial_ct_idx>();
    auto tensor_accessor = TensorAccessor(tensor_args, input_address, page_size);

    constexpr uint32_t scaler = 0;
    generate_reduce_scaler(compute_input_cb_id_1, scaler);

    uint32_t l1_write_addr;
    uint32_t input_granularity_index = 0;

    for (uint32_t tiles_read = start_tiles_read; tiles_read < start_tiles_to_read; tiles_read += num_cores_to_be_used) {
        uint32_t read_tile_id;
        if constexpr (reduction_dim == 0) {
            read_tile_id = tiles_read;
        } else {
            read_tile_id = ((tiles_read / inner_num_tiles) * reduction_num_tiles) + (tiles_read % inner_num_tiles);
        }

        // Now reduce all tiles in the reduction dim. The first index is the
        // same as the output index. After that need to increment by the
        // size of the inner dimensions in tiles. E.g. for 130 tiles,
        // the increment is 130. If 4 tiles need to be reduced, then the
        // first core would access tiles at indices 0, 130, 260, 390, 64,
        // 64+130, 64+260, 64+390, 128, 128+130, 128+260, and 128+390.
        for (uint32_t j = 0; j < reduction_dim_size; ++j) {
            if (input_granularity_index == 0) {
                cb_reserve_back(compute_input_cb_id_0, input_granularity);
                l1_write_addr = get_write_ptr(compute_input_cb_id_0);
            }
            noc_async_read_page(read_tile_id, tensor_accessor, l1_write_addr);

            l1_write_addr += page_size;
            read_tile_id += inner_num_tiles;
            input_granularity_index++;

            if (input_granularity_index == input_granularity) {
                noc_async_read_barrier();
                cb_push_back(compute_input_cb_id_0, input_granularity);
                input_granularity_index = 0;
            }
        }
    }
}
