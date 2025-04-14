// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

static inline unsigned get_tile_id(
    uint32_t i0, uint32_t i1, uint32_t j, uint32_t tiles_per_row, uint32_t PLo, uint32_t PHi, uint32_t HtWt) {
    uint32_t base_tileid = i0 * (tiles_per_row * PHi * HtWt) + i1;
    uint32_t tileid = base_tileid + j * PHi * HtWt;
    return tileid;
}

void kernel_main() {
    // DPRINT << "[Cumsum Writer] start" << ENDL();

    uint32_t output_dram_base_addr = get_arg_val<uint32_t>(0);  // output base addr (DRAM)
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);  // number of tiles in a row / along axis
    uint32_t PHi = get_arg_val<uint32_t>(3);
    uint32_t PLo = get_arg_val<uint32_t>(4);
    uint32_t HtWt = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in = tt::CBIndex::c_1;

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_in);
    uint32_t input_sram_addr = get_read_ptr(cb_in);

    const auto& input_dataformat = get_dataformat(cb_in);
    const auto& output_data_format = get_dataformat(cb_in);

    uint32_t bytes_per_element = 4;
    switch (input_dataformat) {
        case DataFormat::Float32:
        case DataFormat::Int32:
        case DataFormat::UInt32: bytes_per_element = 4; break;
        case DataFormat::Float16_b:
        case DataFormat::Float16: bytes_per_element = 2; break;
        default: bytes_per_element = 4; break;
    }

    uint32_t tile_card = ublock_size_bytes / bytes_per_element;
    const uint32_t output_tile_bytes = ublock_size_bytes;

    InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_dram_base_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    DPRINT << "[Cumsum Writer]: bytes/element = " << bytes_per_element << ENDL();

    for (unsigned i0 = 0; i0 < PLo; i0++) {
        for (unsigned i1 = 0; i1 < PHi * HtWt; i1++) {
            for (unsigned j = 0; j < tiles_per_row; j++) {
                uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);
                // DPRINT << "[Cumsum writer] tileid = " << tileid << ENDL();

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
