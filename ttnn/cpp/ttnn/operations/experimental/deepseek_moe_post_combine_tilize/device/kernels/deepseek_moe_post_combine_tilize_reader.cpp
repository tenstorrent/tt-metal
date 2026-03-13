// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t row_offset = 4;        // TODO: (GR) runtime
    uint32_t bytes_per_row = 4;     // TODO: (GR) compile time
    uint32_t read_byte_offset = 4;  // TODO: (GR) runtime

    constexpr uint32_t tile_height = 32;  // TODO: (GR)

    cb_reserve_back(tilize_input_cb_id, tile_height);
    uint32_t l1_write_addr = get_write_ptr(tilize_input_cb_id);

    uint32_t page_id = row_offset;
    for (uint32_t row = 0; row < tile_height; ++row) {
        uint64_t noc_addr = get_noc_addr(page_id, input_tensor_addr_gen) +
                            read_byte_offset noc_async_read(noc_addr, l1_write_addr, subtoken_size);

        l1_write_addr += bytes_per_row;
        page_id++;
    }

    noc_async_read_barrier();
    cb_push_back(tilize_input_cb_id, tile_height);
}
