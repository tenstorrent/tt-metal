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
    DPRINT << "[Cumsum Reader] start" << ENDL();

    uint32_t input_dram_base_addr = get_arg_val<uint32_t>(0);  // input base addr (DRAM)
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);  // number of tiles in a row / along axis
    uint32_t PHi = get_arg_val<uint32_t>(3);
    uint32_t PLo = get_arg_val<uint32_t>(4);
    uint32_t HtWt = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = tt::CBIndex::c_0;
    constexpr uint32_t cb_zero = tt::CBIndex::c_16;

    const auto& input_data_format = get_dataformat(cb_out);

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_out);
    uint32_t l1_addr_out = get_write_ptr(cb_out);

    const uint32_t input_tile_bytes = ublock_size_bytes;
    const uint32_t output_tile_bytes = ublock_size_bytes;
    InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_dram_base_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    union {
        int32_t i32;
        float f32;
    } scaler;

    uint32_t bytes_per_element = 4;
    switch (input_data_format) {
        case DataFormat::Float32:
            scaler.f32 = 0.f;
            bytes_per_element = 4;
            break;
        case DataFormat::Float16_b:
        case DataFormat::Float16:
            scaler.i32 = 0;  // {bin(0.h), bin(0.h)} == {0x0, 0x0} == 0u32
            bytes_per_element = 2;
            break;
        default:
            scaler.i32 = 0;
            bytes_per_element = 4;
            break;
    }

    uint32_t tile_card = ublock_size_bytes / bytes_per_element;

    // Fill cb_zero with zero
    // TODO: Handle other data types
    cb_reserve_back(cb_zero, 1);
    uint32_t data_zero_addr = get_write_ptr(cb_zero);
    int32_t* data_zero = (int32_t*)data_zero_addr;
    // Dirty method: Write as if uint32_t element: if element size is lower then this writes multiple element per
    // iteration
    for (uint32_t i = 0; i < ublock_size_bytes / sizeof(uint32_t); i++) {
        data_zero[i] = scaler.i32;
    }
    cb_push_back(cb_zero, 1);

    DPRINT << "[Cumsum Reader] #tiles/row = " << tiles_per_row << ", tile size = " << ublock_size_bytes
           << ", Bytes/Element = " << bytes_per_element << ", tile  card = " << tile_card << ", num rows = " << num_rows
           << ", PHi = " << PHi << ", PLo = " << PLo << ", HtWt = " << HtWt << ENDL();

    for (unsigned i0 = 0; i0 < PLo; i0++) {
        for (unsigned i1 = 0; i1 < PHi * HtWt; i1++) {
            for (unsigned j = 0; j < tiles_per_row; j++) {
                uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);

                cb_reserve_back(cb_out, 1);

                // Read tile
                uint32_t data_sram_addr = get_write_ptr(cb_out);
                noc_async_read_tile(tileid, dram_input_addrg, data_sram_addr);
                noc_async_read_barrier();

                // Write tile
                cb_push_back(cb_out, 1);
            }
        }
    }
}
