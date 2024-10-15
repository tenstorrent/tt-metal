// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

inline uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

inline uint32_t row_tile_location(uint32_t row, uint32_t tile_height) {
    // Return the tile id of a row
    return row / tile_height;
}
void kernel_main() {
    // DPRINT << "reader_tile_expand begin" << ENDL();
    std::uint32_t mem_buffer_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t num_rows = get_arg_val<uint32_t>(1);
    std::uint32_t element_per_row = get_arg_val<uint32_t>(2);
    std::uint32_t horz_expand_count = get_arg_val<uint32_t>(3);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t io_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t datasize_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t tile_width = get_compile_time_arg_val(4);
    constexpr uint32_t tile_height = get_compile_time_arg_val(5);

    const uint32_t tilesize = get_tile_size(io_cb_id);

    const uint32_t num_tile_h = div_up(element_per_row, tile_width);
    const uint32_t num_tile_v = div_up(num_rows, tile_height);

    InterleavedAddrGen<src_is_dram> src_generator = {
        .bank_base_address = mem_buffer_src_addr,
        .page_size = tilesize,
    };

    // Scratchpad page size = element_per_row * datasize_bytes
    cb_reserve_back(scratch_cb_id, div_up(horz_expand_count, tile_width));
    auto tmp_buf = get_write_ptr(scratch_cb_id);

    for (uint32_t i = 0; i < num_tile_v; i++) {
        for (uint32_t j = 0; j < num_tile_h; j++) {
            cb_reserve_back(io_cb_id, 1);
            auto l1_buf = get_write_ptr(io_cb_id);

            auto tile_dram_addr = get_noc_addr(i * num_tile_h + j, src_generator);

            // Read the entire tile into scratch buffer
            noc_async_read(tile_dram_addr, l1_buf, tilesize * datasize_bytes);
            noc_async_read_barrier();

            // io buffer contains individual lines
            cb_push_back(io_cb_id, 1);
        }
    }
}
