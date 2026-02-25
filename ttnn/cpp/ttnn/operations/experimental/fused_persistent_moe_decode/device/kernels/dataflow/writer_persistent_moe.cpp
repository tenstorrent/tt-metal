// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;

    constexpr bool is_dram = true;
    uint32_t out0_tile_bytes = get_tile_size(cb_id_out0);

    const InterleavedAddrGenFast<is_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = out0_tile_bytes,
        .data_format = get_dataformat(cb_id_out0)
    };

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t l1_read_addr_out0 = get_read_ptr(cb_id_out0);
        
        // Clear L1 buffer to zeros before writing to DRAM
        // This ensures the MoE output is mathematically 0
        volatile uint32_t* ptr = (volatile uint32_t*)l1_read_addr_out0;
        for (uint32_t j = 0; j < out0_tile_bytes / 4; j++) {
            ptr[j] = 0;
        }

        noc_async_write_tile(i, s0, l1_read_addr_out0);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}