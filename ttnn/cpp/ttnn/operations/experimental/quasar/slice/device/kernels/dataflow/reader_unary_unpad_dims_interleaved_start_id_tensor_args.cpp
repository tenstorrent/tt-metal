// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t tile_width = get_arg(args::tile_width);
    const uint32_t tile_height = get_arg(args::tile_height);

    // num_unpadded_tiles / num_padded_tiles / input_shape are per-dim arrays read by a
    // runtime-varying index, so they arrive as common runtime varargs:
    //   [0, num_dims)            = num_unpadded_tiles
    //   [num_dims, 2*num_dims)   = num_padded_tiles
    //   [2*num_dims, 3*num_dims) = input_shape
    uint32_t num_unpadded_tiles[num_dims];
    uint32_t num_padded_tiles[num_dims];
    uint32_t input_shape_args[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_tiles[j] = get_common_vararg(j);
        num_padded_tiles[j] = get_common_vararg(num_dims + j);
        input_shape_args[j] = get_common_vararg(2 * num_dims + j);
    }

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // id_per_dim is a per-core array advanced by a runtime-varying index → runtime varargs.
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    const auto s0 = TensorAccessor(tensor::in);

    // Create objects for Device 2.0 API
    DataflowBuffer cb_in0(dfb::cb_in);
    DataflowBuffer cb_tensor(dfb::cb_tensor);
    Noc noc;

    // Get tile size from DFB interface
    const uint32_t tile_size = cb_in0.get_entry_size();

    // Create TensorAccessors for start and end tensors
    const auto start_tensor_accessor = TensorAccessor(tensor::start);
    const auto end_tensor_accessor = TensorAccessor(tensor::end);

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
