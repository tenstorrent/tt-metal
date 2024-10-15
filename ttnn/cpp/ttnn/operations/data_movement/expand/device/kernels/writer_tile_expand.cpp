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
    // DPRINT << "writer_tile_expand begin" << ENDL();
    std::uint32_t mem_buffer_dst_addr = get_arg_val<uint32_t>(0);

    std::uint32_t num_rows = get_arg_val<uint32_t>(1);
    std::uint32_t element_per_row = get_arg_val<uint32_t>(2);
    std::uint32_t vert_expand_count = get_arg_val<uint32_t>(3);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t io_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t datasize_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tile_width = get_compile_time_arg_val(3);
    constexpr uint32_t tile_height = get_compile_time_arg_val(4);

    const uint32_t tilesize = get_tile_size(io_cb_id);

    const uint32_t num_tile_h = div_up(element_per_row, tile_width);
    const uint32_t num_tile_v = div_up(num_rows, tile_height);

    InterleavedAddrGen<dst_is_dram> dst_generator = {
        .bank_base_address = mem_buffer_dst_addr,
        .page_size = tilesize,
    };

    // DPRINT << "num_tile_h: " << num_tile_h << " num_tile_v: " << num_tile_v << ENDL();

    for (uint32_t i = 0; i < num_tile_v; i++) {
        for (uint32_t j = 0; j < num_tile_h; j++) {
            cb_wait_front(io_cb_id, 1);
            auto read_ptr = get_read_ptr(io_cb_id);
            for (uint32_t k = 0; k < vert_expand_count; k++) {
                auto tile_pos = k * num_tile_v + i * num_tile_h + j;
                auto noc_addr = get_noc_addr(tile_pos, dst_generator);
                noc_async_write(read_ptr, noc_addr, tilesize * datasize_bytes);
            }
            cb_pop_front(io_cb_id, 1);
        }
    }
}
