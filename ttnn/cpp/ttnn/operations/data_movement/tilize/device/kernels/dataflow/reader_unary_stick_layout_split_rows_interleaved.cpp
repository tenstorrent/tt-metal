// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_height = 32;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_per_block = get_arg_val<uint32_t>(3);
    const uint32_t block_width_size = get_arg_val<uint32_t>(4);
    const uint32_t num_full_blocks_in_row = get_arg_val<uint32_t>(5);
    const uint32_t start_page_id = get_arg_val<uint32_t>(8);

    constexpr uint32_t page_size = get_compile_time_arg_val(0);  // For ND sharded tensors, page size can be < row size.
    constexpr uint32_t num_pages_in_row =
        get_compile_time_arg_val(1);  // For ND-sharded tensors, each row can have multiple pages (pages).
    constexpr uint32_t size_of_valid_data_in_last_page_in_row =
        get_compile_time_arg_val(2);  // For uneven sharding along the width, the last page could contain padding data,
                                      // so we need to specify the size of valid data we want to read in.

    constexpr auto src_tensor_args = TensorAccessorArgs<3>();

    const auto s = TensorAccessor(src_tensor_args, src_addr, page_size);

    uint64_t base_src_noc_addr[tile_height * num_pages_in_row];

    auto read_tiles = [&](const uint32_t& num_tiles) {
        cb_reserve_back(cb_id_in0, num_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t k = 0; k < tile_height; k++) {
            // Need an inner loop for pages within row. Only relevant for ND-sharded case on multicore
            // (otherwise this loop only has 1 iteration).
            for (uint32_t l = 0; l < num_pages_in_row; l++) {
                uint64_t src_noc_addr = base_src_noc_addr[k * num_pages_in_row + l];
                uint32_t width_size =
                    (l == num_pages_in_row - 1) ? size_of_valid_data_in_last_page_in_row : block_width_size;
                noc_async_read(src_noc_addr, l1_write_addr, width_size);
                l1_write_addr += width_size;
                base_src_noc_addr[k * num_pages_in_row + l] += width_size;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_tiles);
    };

    uint32_t page_id = start_page_id;
    for (uint32_t i = 0; i < num_rows / tile_height; i++) {
        // Get Base Addresses
        for (uint32_t j = 0; j < tile_height; j++) {
            for (uint32_t k = 0; k < num_pages_in_row; k++) {
                // For ND-sharded case, we need to read in all pages within the row.
                base_src_noc_addr[j * num_pages_in_row + k] = s.get_noc_addr(page_id);
                page_id++;
            }
        }

        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block);
        }
    }
}
