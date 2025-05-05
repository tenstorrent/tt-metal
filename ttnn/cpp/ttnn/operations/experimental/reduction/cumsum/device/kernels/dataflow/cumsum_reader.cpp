// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "cumsum_common.hpp"

void kernel_main() {
    uint32_t input_dram_base_addr = get_arg_val<uint32_t>(0);  // input base addr (DRAM)
    uint32_t tiles_per_row = get_arg_val<uint32_t>(1);         // number of tiles in a row / along axis
    uint32_t product_high_dims = get_arg_val<uint32_t>(2);
    uint32_t product_low_dims = get_arg_val<uint32_t>(3);
    uint32_t HtWt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = tt::CBIndex::c_0;
    constexpr uint32_t cb_zero = tt::CBIndex::c_2;

    const auto& input_data_format = get_dataformat(cb_out);

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_out);
    uint32_t l1_addr_out = get_write_ptr(cb_out);
    const uint32_t input_tile_bytes = ublock_size_bytes;

    InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_dram_base_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    union {
        float f;
        uint32_t u;
    } scaler;

    const DataFormat data_format = get_dataformat(cb_zero);
    switch ((uint)data_format & 0x1F) {
        case ((uint8_t)DataFormat::Int32):
        case ((uint8_t)DataFormat::UInt32): scaler.u = 0; break;
        case ((uint8_t)DataFormat::Float32):
        case ((uint8_t)DataFormat::Float16_b):
        default: scaler.f = 0.0f; break;
    }

    fill_cb_with_value(cb_zero, scaler.u);

    for (unsigned i0 = 0; i0 < product_low_dims; i0++) {
        for (unsigned i1 = 0; i1 < product_high_dims * HtWt; i1++) {
            for (unsigned j = 0; j < tiles_per_row; j++) {
                uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, product_low_dims, product_high_dims, HtWt);

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
