// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    const uint32_t tile_width = get_arg(args::tile_width);
    const uint32_t tile_height = get_arg(args::tile_height);

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // The host packs three per-dimension blocks into the common vararg channel, in this order:
    // unpadded tile counts, padded tile counts, input shape. These accessors keep the reads below
    // reading the way they did when each block was a separate L1 pointer.
    auto num_unpadded_tiles = [](uint32_t j) { return get_common_vararg(j); };
    auto num_padded_tiles = [](uint32_t j) { return get_common_vararg(num_dims + j); };
    auto input_shape_arg = [](uint32_t j) { return get_common_vararg(2 * num_dims + j); };

    // This node's per-dimension index vector, copied out of the runtime vararg block because the walk
    // at the bottom advances it like an odometer and varargs are read-only values. The host supplies a
    // fresh block on every dispatch, so nothing is carried across enqueues by keeping it local.
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    const auto s0 = TensorAccessor(tensor::src);

    // Create objects for Device 2.0 API
    DataflowBuffer dfb_in0(dfb::in0);
    // Single-entry scratch this kernel stages the index tensors through; it is the only toucher and
    // the host binds it as both endpoints.
    DataflowBuffer dfb_index(dfb::index);
    Noc noc;

    // Get tile size from the DFB entry size
    const uint32_t tile_size = dfb_in0.get_entry_size();

    // Create TensorAccessors for start and end tensors
    const auto start_tensor_accessor = TensorAccessor(tensor::start);
    const auto end_tensor_accessor = TensorAccessor(tensor::end);

    // Read start and end indices from tensors using TensorAccessor
    uint32_t start_indices[num_dims];
    [[maybe_unused]] uint32_t end_indices[num_dims];

    // Read start tensor data using the separate staging buffer
    dfb_index.reserve_back(1);
    uint32_t start_buffer_l1_addr = dfb_index.get_write_ptr();
    noc.async_read(start_tensor_accessor, dfb_index, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    // Complete the producer/consumer handshake (reserve -> push -> wait -> pop) so the scratch buffer
    // is left balanced after this single-tile staging read.
    dfb_index.push_back(1);
    dfb_index.wait_front(1);

    volatile tt_l1_ptr uint32_t* start_data = (volatile tt_l1_ptr uint32_t*)start_buffer_l1_addr;

    for (uint32_t i = 0; i < num_dims; i++) {
        start_indices[i] = start_data[i];
    }
    dfb_index.pop_front(1);

    // Read end tensor data using the separate staging buffer
    dfb_index.reserve_back(1);
    uint32_t end_buffer_l1_addr = dfb_index.get_write_ptr();
    noc.async_read(end_tensor_accessor, dfb_index, tile_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    // Complete the producer/consumer handshake (reserve -> push -> wait -> pop) so the scratch buffer
    // is left balanced after this single-tile staging read.
    dfb_index.push_back(1);
    dfb_index.wait_front(1);

    volatile tt_l1_ptr uint32_t* end_data = (volatile tt_l1_ptr uint32_t*)end_buffer_l1_addr;

    for (uint32_t i = 0; i < num_dims; i++) {
        end_indices[i] = end_data[i];
    }
    dfb_index.pop_front(1);

    uint32_t start_offset = 0;

    if (num_dims >= 2) {
        uint32_t start_h_tiles = start_indices[num_dims - 2] / tile_height;
        uint32_t start_w_tiles = start_indices[num_dims - 1] / tile_width;

        uint32_t input_width = input_shape_arg(num_dims - 1);
        uint32_t input_height = input_shape_arg(num_dims - 2);
        uint32_t num_pages_width = input_width / tile_width;

        start_offset += start_h_tiles * num_pages_width + start_w_tiles;

        if (num_dims > 2) {
            uint32_t upper_dims_offset = 0;
            uint32_t multiplier = (input_height / tile_height) * num_pages_width;

            // Row-major Horner; must match get_upper_start_offset() in slice_device_operation.cpp
            for (uint32_t i = 0; i + 2 < num_dims; ++i) {
                upper_dims_offset = upper_dims_offset * input_shape_arg(i) + start_indices[i];
            }
            start_offset += upper_dims_offset * multiplier;
        }
    }

    // Add the calculated offset to the base start_id
    uint32_t src_tile_id = start_id + start_offset;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint32_t old_src_tile_id = src_tile_id;

        dfb_in0.reserve_back(1);
        noc.async_read(s0, dfb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(1);

        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == num_unpadded_tiles(j)) {
                id_per_dim[j] = 0;
                src_tile_id += num_padded_tiles(j);

            } else {
                break;
            }
        }
    }
}
