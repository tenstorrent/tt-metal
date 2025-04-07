// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // Simplest program possible
    // 1) Only 1 tile
    // 2) float32 elements
    // 3) Along x-axis

    DPRINT << "[Cumsum Reader] start" << ENDL();

    uint32_t src_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t tmp_sram_addr = get_arg_val<uint32_t>(2);
    uint32_t src_bank_id = get_arg_val<uint32_t>(3);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);
    uint32_t dim = get_arg_val<uint32_t>(6);
    uint32_t tensor_dim = get_arg_val<uint32_t>(7);

    // Load DRAM data into SRAM
    uint32_t src_dram_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_dram_addr);
    uint32_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_dram_addr);

    constexpr uint32_t cb_out = tt::CBIndex::c_0;

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_out);

    uint32_t l1_addr_in = get_write_ptr(cb_out);
    uint32_t tile_card = ublock_size_bytes / sizeof(uint32_t);

    float sum = 0.f;

    // TODO: add `dim`/`how` argument + tensor dimension
    // 2 parameters: tile_offset, and <TODO>

    // If tile-based increment => keep previous tile in memory
    // data[i] += previous[i]
    // previous[i] = data[i]

    // Initialize accumulator buffer
    float* tmp_accumulator = (float*)tmp_sram_addr;
    for (uint32_t i = 0; i < tile_card; i++) {
        tmp_accumulator[i] = 0.f;
    }

    if (dim == tensor_dim - 1) {  // x-axis
        for (uint32_t tile = 0; tile < num_tiles; tile++) {
            // Read DRAM => SRAM
            noc_async_read(src_dram_noc_addr + tile * ublock_size_bytes, l1_addr_in, ublock_size_bytes);
            noc_async_read_barrier();

            float* data = (float*)l1_addr_in;
            DPRINT_DATA0(DPRINT << "[Cumsum reader] data[0] = " << data[0] << ENDL());

            for (uint32_t i = 0; i < tile_card; i++) {
                sum += data[i];
                data[i] = sum;
            }

            noc_async_write(l1_addr_in, dst_dram_noc_addr + tile * ublock_size_bytes, ublock_size_bytes);
            noc_async_write_barrier();

            DPRINT_DATA0(DPRINT << "[Cumsum reader] completed = " << sum << ENDL());
        }
    }
}
