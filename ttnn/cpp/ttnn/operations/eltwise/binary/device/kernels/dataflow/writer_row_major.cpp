// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include <cstdint>

constexpr uint32_t TILE_SIZE = 1024;

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_rows = get_arg_val<uint32_t>(1);
    uint32_t start_row_id = get_arg_val<uint32_t>(2);
    uint32_t row_size = get_arg_val<uint32_t>(3);

    constexpr uint32_t datum_size_bytes = 2;

    // The circular buffer to write the results into
    constexpr uint32_t cb_out0 = tt::CBIndex::c_2;

    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    const InterleavedAddrGen<true> c = {
        .bank_base_address = c_addr, .page_size = datum_size_bytes * row_size};

    // Calculate the range of rows this core should process
    const uint32_t end_row_id = start_row_id + n_rows;
    const uint32_t num_tiles_per_row = (row_size + TILE_SIZE - 1) / TILE_SIZE;

    // Now we loop over the assigned rows and read them into the circular
    // buffers
    for (uint32_t i = start_row_id; i < end_row_id; i++) {
        uint32_t offset = 0;
        for (uint32_t j = 0; j < num_tiles_per_row; ++j) {
            uint64_t c_noc_addr = get_noc_addr(i, c, offset);

            cb_wait_front(cb_out0, 1);
            uint32_t cb_out0_addr = get_write_ptr(cb_out0);

            uint32_t bytes_to_write = min(TILE_SIZE * datum_size_bytes, row_size * datum_size_bytes - offset);
            noc_async_write(cb_out0_addr, c_noc_addr, bytes_to_write);
            noc_async_write_barrier();

            cb_pop_front(cb_out0, 1);

            offset += bytes_to_write;
	    }
    }
}
