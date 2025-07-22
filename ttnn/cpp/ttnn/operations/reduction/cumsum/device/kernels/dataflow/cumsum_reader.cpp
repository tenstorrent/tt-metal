// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "cumsum_common.hpp"

void kernel_main() {
    uint32_t input_dram_base_addr = get_arg_val<uint32_t>(0);  // input base addr (DRAM)
    uint32_t start_row = get_arg_val<uint32_t>(1);
    uint32_t num_rows = get_arg_val<uint32_t>(2);
    uint32_t start_high_tile_index = get_arg_val<uint32_t>(3);
    uint32_t start_low_tile_index = get_arg_val<uint32_t>(4);

    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(0);
    constexpr uint32_t HtWt = get_compile_time_arg_val(1);
    constexpr uint32_t product_high_dims = get_compile_time_arg_val(2);
    constexpr uint32_t product_low_dims = get_compile_time_arg_val(3);
    constexpr uint32_t flip = get_compile_time_arg_val(4);
    constexpr bool is_dram = get_compile_time_arg_val(5) == 1;

    constexpr uint32_t cb_out = tt::CBIndex::c_0;
    constexpr uint32_t cb_zero = tt::CBIndex::c_2;

    const auto& input_data_format = get_dataformat(cb_out);

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_out);
    uint32_t l1_addr_out = get_write_ptr(cb_out);
    const uint32_t input_tile_bytes = ublock_size_bytes;

    InterleavedAddrGenFast<is_dram> dram_input_addrg = {
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

    uint32_t i0 = start_low_tile_index;
    uint32_t i1 = start_high_tile_index;
    for (uint32_t i = start_row; i < start_row + num_rows; i++) {
        for (uint32_t j = 0; j < tiles_per_row; j++) {
            uint32_t tile_j = j;
            if constexpr (flip) {
                tile_j = tiles_per_row - j - 1;
            }

            uint32_t tileid = get_tile_id(i0, i1, tile_j, tiles_per_row, product_low_dims, product_high_dims, HtWt);

            cb_reserve_back(cb_out, 1);

            // Read tile
            uint32_t data_sram_addr = get_write_ptr(cb_out);
            noc_async_read_tile(tileid, dram_input_addrg, data_sram_addr);
            noc_async_read_barrier();

            // Write tile
            cb_push_back(cb_out, 1);
        }

        // The following is equivalent to the following, but does not use integer division
        // i0 = i / (product_high_dims * HtWt);
        // i1 = i % (product_high_dims * HtWt);
        i1++;
        if (i1 >= product_high_dims * HtWt) {
            i1 = 0;
            i0++;
        }
    }
}
