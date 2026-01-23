// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t num_pages_per_tile_row = get_compile_time_arg_val(2);

    constexpr auto dst_args = TensorAccessorArgs<3>();
    const auto s0 = TensorAccessor(dst_args, dst_addr, page_size);

    const uint32_t end_id = start_page_id + num_pages;

    DPRINT << "[WRITER] start " << start_page_id << " end " << end_id << ENDL();
    DPRINT << "[WRITER] pages/row " << num_pages_per_tile_row << ENDL();
    DPRINT << "[WRITER] page size " << page_size << ENDL();

    // reader copied the data from DRAM to CB buffer.
    for (uint32_t i = start_page_id; i < end_id; i += num_pages_per_tile_row) {
        cb_wait_front(cb_id_out, num_pages_per_tile_row);

        DPRINT << "[WRITER] CB " << cb_id_out << ENDL();
        DPRINT << "[WRITER] pages per row " << num_pages_per_tile_row << ENDL();

        uint64_t base_l1_read_addr = get_read_ptr(cb_id_out);

        // Write each tile individually to its correct NOC address
        for (uint32_t j = 0; j < num_pages_per_tile_row; ++j) {
            uint64_t l1_read_addr = base_l1_read_addr + j * page_size;
            // pages in DRAM are not necessarily contiguous since memory is interleaved.
            // that's why we need to get NOC address for each page individually.
            uint64_t dst_noc_addr = s0.get_noc_addr(i + j);

            DPRINT << "[WRITER] Tile " << j << " L1 addr " << HEX() << "0x" << l1_read_addr << DEC() << ENDL();
            DPRINT << "[WRITER] Tile " << j << " DST NOC addr " << HEX() << "0x" << dst_noc_addr << DEC() << ENDL();

            noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        }

        noc_async_write_barrier();

        cb_pop_front(cb_id_out, num_pages_per_tile_row);
    }
}
