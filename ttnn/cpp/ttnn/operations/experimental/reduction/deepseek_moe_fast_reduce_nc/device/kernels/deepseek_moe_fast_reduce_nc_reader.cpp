// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

inline uint32_t get_read_tile_id(uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size) {
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
}

constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr uint32_t input_granularity = get_compile_time_arg_val(1);
constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(2);
constexpr uint32_t reduction_dim = get_compile_time_arg_val(3);
constexpr uint32_t compute_input_cb_id_0 = get_compile_time_arg_val(4);
constexpr uint32_t compute_input_cb_id_1 = get_compile_time_arg_val(5);

constexpr uint32_t initial_ct_idx = 6;

void kernel_main() {
    uint32_t arg_idx = 0;

    uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_input_tiles = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t id_range_length = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t reduce_tile_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t inner_tile_size = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto tensor_args = TensorAccessorArgs<initial_ct_idx>();
    auto tensor_accessor = TensorAccessor(tensor_args, input_address, page_size);

    constexpr uint32_t scaler = 0;
    generate_reduce_scaler(compute_input_cb_id_1, scaler);

    uint32_t l1_write_addr_in0;
    uint32_t input_granularity_index = 0;

    // For each shard, start at the index of the first shard to be reduced (same
    // index as output), then increment by the appropriate increment (based on
    // the grid size), until the range length is reached. E.g. For 130 shards
    // on an 8x8 grid, the first core would have start_id equal 0,
    // num_cores_to_be_used equal 64, and id_range_length 64*3. The outer_id
    // values would be 0, 64, and 128.
    for (uint32_t outer_id = start_id; outer_id < start_id + id_range_length; outer_id += num_cores_to_be_used) {
        uint32_t read_tile_id;
        if constexpr (reduction_dim == 0) {
            read_tile_id = outer_id;
        } else {
            read_tile_id = get_read_tile_id(outer_id, reduce_tile_size, inner_tile_size);
        }

        // Now reduce all tiles in the reduction dim. The first index is the
        // same as the output index. After that need to increment by the
        // size of the inner dimensions in tiles. E.g. for 130 tiles
        // (where shard factor equals 1), the increment is 130. If 4 tiles
        // need to be reduced, then the first core would access tiles at
        // indices 0, 130, 260, 390, 64, 64+130, 64+260, 64+390, 128,
        // 128+130, 128+260, and 128+390.
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            if (input_granularity_index == 0) {
                cb_reserve_back(compute_input_cb_id_0, input_granularity);
                l1_write_addr_in0 = get_write_ptr(compute_input_cb_id_0);
            }

            uint64_t noc_read_addr = get_noc_addr(read_tile_id, tensor_accessor);
            noc_async_read(noc_read_addr, l1_write_addr_in0, page_size);

            l1_write_addr_in0 += page_size;
            read_tile_id += inner_tile_size;
            input_granularity_index++;

            if (input_granularity_index == input_granularity) {
                noc_async_read_barrier();
                cb_push_back(compute_input_cb_id_0, input_granularity);
                input_granularity_index = 0;
            }
        }
    }
}
