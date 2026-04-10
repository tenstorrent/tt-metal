// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_per_block = get_arg_val<uint32_t>(3);
    const uint32_t block_width_size = get_arg_val<uint32_t>(4);
    const uint32_t num_full_blocks_in_row = get_arg_val<uint32_t>(5);
    const uint32_t start_page_id = get_arg_val<uint32_t>(8);

    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages_per_block = get_compile_time_arg_val(1);
    constexpr uint32_t size_of_valid_data_in_last_page = get_compile_time_arg_val(2);
    constexpr uint32_t total_pages_per_row = get_compile_time_arg_val(3);

    constexpr auto src_tensor_args = TensorAccessorArgs<4>();

    const auto s = TensorAccessor(src_tensor_args, src_addr, page_size);

    auto read_tiles = [&](const uint32_t& num_tiles, uint32_t page_id) {
        cb_reserve_back(cb_id_in0, num_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t k = 0; k < tile_height; k++) {
            for (uint32_t l = 0; l < num_pages_per_block; l++) {
                uint64_t src_noc_addr = static_cast<uint64_t>(s.get_noc_addr(page_id + l));
                uint32_t width_size =
                    (l == num_pages_per_block - 1) ? size_of_valid_data_in_last_page : block_width_size;
                noc_async_read(src_noc_addr, l1_write_addr, width_size);
                l1_write_addr += width_size;
            }
            page_id += total_pages_per_row;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_tiles);
    };

    uint32_t page_id = start_page_id;
    for (uint32_t i = 0; i < num_rows / tile_height; i++) {
        uint32_t block_page_id = page_id;
        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, block_page_id);
            block_page_id += num_pages_per_block;
        }
        page_id += tile_height * total_pages_per_row;
    }
}
