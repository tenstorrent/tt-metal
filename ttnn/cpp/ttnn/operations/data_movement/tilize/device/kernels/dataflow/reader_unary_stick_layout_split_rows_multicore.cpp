// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_per_block = get_arg_val<uint32_t>(3);
    const uint32_t block_width_size = get_arg_val<uint32_t>(4);
    const uint32_t num_full_blocks_in_row = get_arg_val<uint32_t>(5);
    const uint32_t start_page_id = get_arg_val<uint32_t>(8);

    constexpr uint32_t num_pages_in_row =
        get_compile_time_arg_val(1);  // For ND-sharded tensors, each row can have multiple pages.
    constexpr uint32_t size_of_valid_data_in_last_page_in_row =
        get_compile_time_arg_val(2);  // For uneven sharding along the width, the last page could contain padding data,
                                      // so we need to specify the size of valid data we want to read in.

    constexpr auto src_tensor_args = TensorAccessorArgs<3>();

    const auto s = TensorAccessor(src_tensor_args, src_addr);

    experimental::CircularBuffer cb(cb_id_in0);
    experimental::Noc noc;

    auto read_tiles = [&](const uint32_t& num_tiles, uint32_t page_id) {
        cb.reserve_back(num_tiles);
        uint32_t l1_write_offset = 0;
        for (uint32_t k = 0; k < tile_height; k++) {
            // Need an inner loop for pages within row. Only relevant for ND-sharded case on multicore
            // (otherwise this loop only has 1 iteration).
            for (uint32_t l = 0; l < num_pages_in_row; l++) {
                uint32_t width_size =
                    (l == num_pages_in_row - 1) ? size_of_valid_data_in_last_page_in_row : block_width_size;
                noc.async_read(
                    s, cb, width_size, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = l1_write_offset});
                page_id++;
                l1_write_offset += width_size;
            }
        }
        noc.async_read_barrier();
        cb.push_back(num_tiles);
    };

    uint32_t page_id = start_page_id;
    for (uint32_t i = 0; i < num_rows / tile_height; i++) {
        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, page_id);
        }
        page_id += tile_height * num_pages_in_row;
    }
}
