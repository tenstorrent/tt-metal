// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t shard_factor = get_compile_time_arg_val(0);
    constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(1);
    constexpr uint32_t outer_id_increment = shard_factor * num_cores_to_be_used;
    constexpr uint32_t left_width = get_compile_time_arg_val(2);  // zero disables splitting
    constexpr uint32_t full_width = get_compile_time_arg_val(3);
    constexpr auto tensor_args = TensorAccessorArgs<4>();

    // runtime args
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto id_range_length = get_arg_val<uint32_t>(1);
    const auto start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;

    Noc noc;
    CircularBuffer cb_out_obj(cb_id_out);

    uint32_t output_tile_bytes = get_tile_size(cb_id_out);

    auto tensor_accessor = TensorAccessor(tensor_args, output_addr);

    const auto right_accessor = [&] {
        if constexpr (left_width > 0) {
            constexpr auto right_args = TensorAccessorArgs<tensor_args.next_compile_time_args_offset()>();
            return TensorAccessor(right_args, get_arg_val<uint32_t>(3));
        } else {
            return tensor_accessor;
        }
    }();

    // For each shard, start at the index of the first shard to be reduced (same
    // index as output), then increment by the appropriate increment (based on
    // the grid size), until the range length is reached. See reader and program
    // factory for examples.
    for (uint32_t outer_id = start_id; outer_id < start_id + id_range_length; outer_id += outer_id_increment) {
        for (uint32_t id_offset = 0; id_offset < shard_factor; id_offset++) {
            uint32_t i = outer_id + id_offset;
            uint32_t write_tile_id = i;
            cb_out_obj.wait_front(onetile);
            if constexpr (left_width > 0) {
                constexpr uint32_t right_width = full_width - left_width;
                const uint32_t row = i / full_width;
                const uint32_t col = i % full_width;
                if (col < left_width) {
                    noc.async_write(
                        cb_out_obj,
                        tensor_accessor,
                        output_tile_bytes,
                        {.offset_bytes = 0},
                        {.page_id = row * left_width + col});
                } else {
                    noc.async_write(
                        cb_out_obj,
                        right_accessor,
                        output_tile_bytes,
                        {.offset_bytes = 0},
                        {.page_id = row * right_width + col - left_width});
                }
            } else {
                noc.async_write(
                    cb_out_obj, tensor_accessor, output_tile_bytes, {.offset_bytes = 0}, {.page_id = write_tile_id});
            }
            noc.async_write_barrier();
            cb_out_obj.pop_front(onetile);
        }
    }
}
