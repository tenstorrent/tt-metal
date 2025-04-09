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

    uint32_t input_dram_base_addr = get_arg_val<uint32_t>(0);   // input base addr (DRAM)
    uint32_t output_dram_base_addr = get_arg_val<uint32_t>(1);  // output base addr (DRAM)
    uint32_t tmp_sram_addr = get_arg_val<uint32_t>(2);          // accumulator tile addr (SRAM)
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(4);  // number of tiles in a row / along axis
    uint32_t PHi = get_arg_val<uint32_t>(5);
    uint32_t PLo = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);

    // Load DRAM data into SRAM
    // uint32_t src_dram_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, input_dram_base_addr);
    // uint32_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, output_dram_base_addr);

    constexpr uint32_t cb_out = tt::CBIndex::c_0;

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_out);

    uint32_t l1_addr_in = get_write_ptr(cb_out);
    uint32_t tile_card = ublock_size_bytes / sizeof(float);

    float sum = 0.f;

    // If tile-based increment => keep previous tile in memory
    // data[i] += previous[i]
    // previous[i] = data[i]

    // Set SRAM addresses
    float* const accumulator = (float*)tmp_sram_addr;
    float* const data = (float*)l1_addr_in;

    DPRINT << "[Cumsum Reader] #tiles/row = " << tiles_per_row << ", tile size = " << ublock_size_bytes
           << ", tile  card = " << tile_card << ", num rows = " << num_rows << ", PHi = " << PHi << ", PLo = " << PLo
           << ", HtWt = " << HtWt << ENDL();

    const uint32_t input_tile_bytes = ublock_size_bytes;
    const uint32_t output_tile_bytes = ublock_size_bytes;
    const auto& input_data_format = get_dataformat(cb_out);  // Note: we don't use CB for now so that's OK
    const auto& output_data_format = get_dataformat(cb_out);

    InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_dram_base_addr, .page_size = input_tile_bytes, .data_format = input_data_format};
    InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_dram_base_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    for (unsigned i0 = 0; i0 < PLo; i0++) {
        for (unsigned i1 = 0; i1 < PHi * HtWt; i1++) {
            // Set accumulator to 0
            for (unsigned i = 0; i < tile_card; i++) {
                accumulator[i] = 0.f;
            }

            for (unsigned j = 0; j < tiles_per_row; j++) {
                uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);
                DPRINT << "[Cumsum Reader] tile = " << tileid << ENDL();

                // Read tile
                noc_async_read_tile(tileid, dram_input_addrg, l1_addr_in);
                noc_async_read_barrier();

                // Accumulate within tile
                for (unsigned i = 0; i < tile_card; i++) {
                    accumulator[i] += data[i];
                }

                // Write tile
                noc_async_write_tile(tileid, dram_output_addrg, tmp_sram_addr);
                noc_async_write_barrier();
            }
        }
    }

    /*for (uint32_t tile_row = 0; tile_row < num_rows; tile_row++) {
        const uint32_t addr_row_offset = tile_row * tile_stride * num_tiles_per_row * sizeof(float);
        uint32_t input_row_dram_addr = input_dram_base_addr + addr_row_offset;
        uint32_t output_row_dram_addr = output_dram_base_addr + addr_row_offset;


        DPRINT << "[Cumsum Reader] input base dram addr = " << input_dram_base_addr
                << ", output base dram addr = " << output_dram_base_addr
                << ", input row dram addr = " << input_row_dram_addr
                << ", output row dram addr = " << output_row_dram_addr
                << ENDL();

        // Set accumulator tile to 0
        for (uint32_t i = 0; i < tile_card; i++) {
            tmp_accumulator[i] = 0.f;
        }

        for (uint32_t tile = 0; tile < low_dims; tile++) {
            const uint32_t local_offset = tile * tile_stride * sizeof(float);
            uint32_t input_dram_addr = input_row_dram_addr + local_offset;
            uint32_t output_dram_addr = output_row_dram_addr + local_offset;

            uint32_t tile_id = tile_row * low_dims + tile * high_dims;
            DPRINT << "[Cumsum Reader] tile id = " << tile_id
                    << ", offset = " << local_offset + addr_row_offset << ENDL();

            // Read DRAM => SRAM
            noc_async_read_tile(tile_id, dram_input_addrg, l1_addr_in);
            noc_async_read_barrier();

            for (uint32_t i = 0; i < tile_card; i++) {
                tmp_accumulator[i] += data[i];
            }

            noc_async_write_tile(tile_id, dram_output_addrg, tmp_sram_addr);
            noc_async_write_barrier();
        }
    }*/
}
