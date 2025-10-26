// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t shard_factor = get_compile_time_arg_val(0);
    constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(1);
    constexpr uint32_t outer_id_increment = shard_factor * num_cores_to_be_used;
    constexpr auto tensor_args = TensorAccessorArgs<2>();

    // runtime args
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto id_range_length = get_arg_val<uint32_t>(1);
    const auto start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;

    uint32_t output_tile_bytes = get_tile_size(cb_id_out);

    auto tensor_accessor = TensorAccessor(tensor_args, output_addr, output_tile_bytes);

    // For each shard, start at the index of the first shard to be reduced (same
    // index as output), then increment by the appropriate increment (based on
    // the grid size), until the range length is reached. See reader and program
    // factory for examples.
    for (uint32_t outer_id = start_id; outer_id < start_id + id_range_length; outer_id += outer_id_increment) {
        for (uint32_t id_offset = 0; id_offset < shard_factor; id_offset++) {
            uint32_t i = outer_id + id_offset;
            uint32_t write_tile_id = i;
            cb_wait_front(cb_id_out, onetile);

            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_page(write_tile_id, tensor_accessor, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, onetile);
        }
    }
}
