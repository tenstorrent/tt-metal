// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_tensor = get_compile_time_arg_val(1);
    constexpr uint32_t num_dims = get_compile_time_arg_val(2);
    const uint32_t tile_width = get_compile_time_arg_val(3);
    const uint32_t tile_height = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto start_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t end_addr = get_common_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* num_unpadded_tiles = (volatile tt_l1_ptr uint32_t*)(get_common_arg_addr(3));
    volatile tt_l1_ptr uint32_t* num_padded_tiles = num_unpadded_tiles + num_dims;

    const uint32_t start_id = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    tt_l1_ptr uint32_t* id_per_dim = (tt_l1_ptr uint32_t*)(get_arg_addr(2));

    constexpr uint32_t tile_size = get_tile_size(cb_id_in0);

    const auto s0 = TensorAccessor(src_args, src_addr, tile_size);

    // Create experimental objects for Device 2.0 API
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_tensor(cb_id_tensor);
    experimental::Noc noc;

    // Create TensorAccessors for start and end tensors
    const auto start_tensor_accessor = TensorAccessor(start_args, start_addr, tile_size);
    const auto end_tensor_accessor = TensorAccessor(end_args, end_addr, tile_size);

    // Read start and end indices from tensors using TensorAccessor
    uint32_t start_indices[num_dims];
    uint32_t end_indices[num_dims];

    // Read start tensor data using separate circular buffer
    cb_tensor.reserve_back(1);
    uint32_t start_buffer_l1_addr = cb_tensor.get_write_ptr();
    noc.async_read(start_tensor_accessor, cb_tensor, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* start_data = (volatile tt_l1_ptr uint32_t*)start_buffer_l1_addr;

    for (uint32_t i = 0; i < num_dims; i++) {
        start_indices[i] = start_data[i];
    }
    cb_tensor.pop_front(1);

    // Read end tensor data using separate circular buffer
    cb_tensor.reserve_back(1);
    uint32_t end_buffer_l1_addr = cb_tensor.get_write_ptr();
    noc.async_read(end_tensor_accessor, cb_tensor, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* end_data = (volatile tt_l1_ptr uint32_t*)end_buffer_l1_addr;

    for (uint32_t i = 0; i < num_dims; i++) {
        end_indices[i] = end_data[i];
    }
    cb_tensor.pop_front(1);

    uint32_t start_offset = 0;

    if (num_dims >= 2) {
        uint32_t start_h_tiles = start_indices[num_dims - 2] / tile_height;
        uint32_t start_w_tiles = start_indices[num_dims - 1] / tile_width;

        volatile tt_l1_ptr uint32_t* input_shape_args =
            (volatile tt_l1_ptr uint32_t*)(get_common_arg_addr(3 + 2 * num_dims));
        uint32_t input_width = input_shape_args[num_dims - 1];
        uint32_t input_height = input_shape_args[num_dims - 2];
        uint32_t num_pages_width = input_width / tile_width;

        start_offset += start_h_tiles * num_pages_width + start_w_tiles;

        if (num_dims > 2) {
            uint32_t upper_dims_offset = 0;
            uint32_t multiplier = (input_height / tile_height) * num_pages_width;

            for (int32_t i = num_dims - 3; i >= 0; i--) {
                upper_dims_offset = upper_dims_offset * input_shape_args[i] + start_indices[i];
            }
            start_offset += upper_dims_offset * multiplier;
        }
    }

    // Add the calculated offset to the base start_id
    uint32_t src_tile_id = start_id + start_offset;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint32_t old_src_tile_id = src_tile_id;

        cb_in0.reserve_back(1);
        noc.async_read(s0, cb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(1);

        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == num_unpadded_tiles[j]) {
                id_per_dim[j] = 0;
                src_tile_id += num_padded_tiles[j];

            } else {
                break;
            }
        }
    }
}
