// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

inline uint32_t get_read_tile_id(uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size) {
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
}

void kernel_main() {
    // compile-time args
    constexpr uint32_t input_granularity = get_compile_time_arg_val(0);
    constexpr uint32_t shard_factor = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(2);
    constexpr uint32_t outer_id_increment = shard_factor * num_cores_to_be_used;
    constexpr auto tensor_args = TensorAccessorArgs<3>();

    // runtime args
    const auto input_addr = get_arg_val<uint32_t>(0);
    const auto num_input_tiles = get_arg_val<uint32_t>(1);
    const auto id_range_length = get_arg_val<uint32_t>(2);
    const auto start_id = get_arg_val<uint32_t>(3);
    const auto dim = get_arg_val<uint32_t>(4);
    const auto reduce_tile_size = get_arg_val<uint32_t>(5);
    const auto inner_tile_size = get_arg_val<uint32_t>(6);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t scaler = 0;

    generate_reduce_scaler(cb_id_in1, scaler);

    uint32_t l1_write_addr_in0;
    constexpr uint32_t input_tile_bytes = get_tile_size(cb_id_in0);

    auto tensor_accessor = TensorAccessor(tensor_args, input_addr, input_tile_bytes);
    uint32_t input_granularity_index = 0;

    // For each shard, start at the index of the first shard to be reduced (same
    // index as output), then increment by the appropriate increment (based on
    // the grid size), until the range length is reached. E.g. For 130 shards
    // on an 8x8 grid, the first core would have start_id equal 0,
    // outer_id_increment equal 64, and id_range_length 64*3. The outer_id
    // values would be 0, 64, and 128.
    for (uint32_t outer_id = start_id; outer_id < start_id + id_range_length; outer_id += outer_id_increment) {
        // Go through each tile of each shard.
        for (uint32_t id_offset = 0; id_offset < shard_factor; ++id_offset) {
            uint32_t i = outer_id + id_offset;
            auto read_tile_id = (dim == 0) ? (i) : (get_read_tile_id(i, reduce_tile_size, inner_tile_size));
            // Now reduce all tiles in the reduction dim. The first index is the
            // same as the output index. After that need to increment by the
            // size of the inner dimensions in tiles. E.g. for 130 tiles
            // (where shard factor equals 1), the increment is 130. If 4 tiles
            // need to be reduced, then the first core would access tiles at
            // indices 0, 130, 260, 390, 64, 64+130, 64+260, 64+390, 128,
            // 128+130, 128+260, and 128+390.
            for (uint32_t j = 0; j < num_input_tiles; ++j) {
                if (input_granularity_index == 0) {
                    cb_reserve_back(cb_id_in0, input_granularity);
                    l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                }
                noc_async_read_tile(read_tile_id, tensor_accessor, l1_write_addr_in0);
                l1_write_addr_in0 += input_tile_bytes;
                read_tile_id += inner_tile_size;
                input_granularity_index++;
                if (input_granularity_index == input_granularity) {
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in0, input_granularity);
                    input_granularity_index = 0;
                }
            }
        }
    }
}
