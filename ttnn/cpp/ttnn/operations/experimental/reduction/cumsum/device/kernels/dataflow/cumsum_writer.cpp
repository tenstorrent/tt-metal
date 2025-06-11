// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

#include "cumsum_common.hpp"

void kernel_main() {
    uint32_t output_dram_base_addr = get_arg_val<uint32_t>(0);  // output base addr (DRAM)
    uint32_t tiles_per_row = get_arg_val<uint32_t>(1);          // number of tiles in a row / along axis
    uint32_t product_high_dims = get_arg_val<uint32_t>(2);
    uint32_t product_low_dims = get_arg_val<uint32_t>(3);
    uint32_t HtWt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = tt::CBIndex::c_1;

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_in);
    uint32_t input_sram_addr = get_read_ptr(cb_in);

    const auto& input_dataformat = get_dataformat(cb_in);
    const auto& output_data_format = get_dataformat(cb_in);

    const uint32_t output_tile_bytes = ublock_size_bytes;

    InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_dram_base_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    for (unsigned i0 = 0; i0 < product_low_dims; i0++) {
        for (unsigned i1 = 0; i1 < product_high_dims * HtWt; i1++) {
            for (unsigned j = 0; j < tiles_per_row; j++) {
                uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, product_low_dims, product_high_dims, HtWt);

                // Read tile from Circularbuffer
                cb_wait_front(cb_in, 1);

                // Write tile
                noc_async_write_tile(tileid, dram_output_addrg, input_sram_addr);
                noc_async_write_barrier();

                cb_pop_front(cb_in, 1);
            }
        }
    }
}
