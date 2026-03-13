// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t output_tile_offset = 4;  // TODO: (GR) runtime
    uint32_t tile_page_size = 4;      // TODO: (GR) compile time
    uint32_t num_tiles = 4;           // TODO: (GR) compile time

    cb_wait_front(tilize_output_cb_id, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(tilize_output_cb_id);

    uint32_t page_id = output_tile_offset;
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint64_t noc_addr =
            get_noc_addr(page_id, output_tensor_addr_gen) noc_async_write(noc_addr, l1_read_addr, tile_page_size)

                l1_read_addr += tile_page_size;
        page_id++;
    }

    noc_async_write_barrier();
    cb_pop_front(tilize_output_cb_id, num_tiles);
}
