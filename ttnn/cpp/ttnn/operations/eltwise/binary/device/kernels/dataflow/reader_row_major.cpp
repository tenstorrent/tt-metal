// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include <cstdint>

constexpr uint32_t TILE_SIZE = 1024;

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t n_rows = get_arg_val<uint32_t>(2);
    uint32_t start_row_id = get_arg_val<uint32_t>(3);
    uint32_t row_size = get_arg_val<uint32_t>(7);

    constexpr uint32_t datum_size_bytes = 2;

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    const InterleavedAddrGen<true> a = {
        .bank_base_address = a_addr, .page_size = datum_size_bytes * row_size};

    const InterleavedAddrGen<true> b = {
        .bank_base_address = b_addr, .page_size = datum_size_bytes * row_size};

    // Calculate the range of rows this core should process
    const uint32_t end_row_id = start_row_id + n_rows;
    const uint32_t num_tiles_per_row = (row_size + TILE_SIZE - 1) / TILE_SIZE;

    // Now we loop over the assigned rows and read them into the circular
    // buffers
    for (uint32_t i = start_row_id; i < end_row_id; i++) {
        uint32_t offset = 0;
        for (uint32_t j = 0; j < num_tiles_per_row; ++j) {
            uint64_t a_noc_addr = get_noc_addr(i, a, offset);
            uint64_t b_noc_addr = get_noc_addr(i, b, offset);

            cb_reserve_back(cb_in0, 1);
            uint32_t cb_in0_addr = get_write_ptr(cb_in0);
            cb_reserve_back(cb_in1, 1);
            uint32_t cb_in1_addr = get_write_ptr(cb_in1);

            uint32_t bytes_to_read = min(TILE_SIZE * datum_size_bytes, row_size * datum_size_bytes - offset);
	        noc_async_read(a_noc_addr, cb_in0_addr, bytes_to_read);
	        noc_async_read(b_noc_addr, cb_in1_addr, bytes_to_read);

            noc_async_read_barrier();

            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);

            offset += bytes_to_read;
	    }
    }
}
