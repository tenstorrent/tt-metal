// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint_pages.h"

constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t ELEMENT_SIZE_BYTES = 2;
constexpr uint32_t STICK_SIZE = TILE_SIZE * ELEMENT_SIZE_BYTES;

void kernel_main() {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in_transpose = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    const uint32_t channel_size = C * ELEMENT_SIZE_BYTES;

    cb_reserve_back(cb_out, 1);

    const uint32_t base_l1_write_addr = get_write_ptr(cb_out);
    DPRINT << "cbs " << cb_in_transpose << " " << cb_out << " C=" << C << ENDL();

    uint32_t write_addr = base_l1_write_addr;
    for (uint32_t i = 0; i < total_tiles; i++) {
        DPRINT << "tile=" << i << ENDL();
        cb_wait_front(cb_in_transpose, 1);

        uint64_t base_l1_read_addr = get_noc_addr(get_read_ptr(cb_in_transpose));
        uint64_t l1_read_addr = base_l1_read_addr;
        for (uint32_t row = 0; row < TILE_SIZE; row++) {
            DPRINT << "row=" << row << ENDL();
            DPRINT << "read=" << l1_read_addr - base_l1_read_addr << " write=" << write_addr - base_l1_write_addr
                   << " size=" << channel_size << ENDL();
            noc_async_read(l1_read_addr, write_addr, channel_size);
            l1_read_addr += STICK_SIZE;
            write_addr += channel_size;
        }
        cb_pop_front(cb_in_transpose, 1);
    }

    DPRINT << "done writer" << ENDL();
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);
}
