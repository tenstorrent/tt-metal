// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../cumprod_common.hpp"

void kernel_main() {
    uint32_t output_base_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    uint32_t input_tile_offset = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    uint32_t is_output_dram = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = tt::CBIndex::c_1;

    uint32_t ublock_size_bytes = get_tile_size(cb_out);
    uint32_t input_sram_addr = get_read_ptr(cb_out);

    const auto& input_dataformat = get_dataformat(cb_out);
    const auto& output_data_format = get_dataformat(cb_out);

    uint32_t bytes_per_element = 4;

    switch (input_dataformat) {
        case DataFormat::Float32: bytes_per_element = 4; break;
        case DataFormat::Float16_b:
        case DataFormat::Float16: bytes_per_element = 2; break;
        case DataFormat::UInt8: bytes_per_element = 1; break;
        case DataFormat::UInt16: bytes_per_element = 2; break;
        case DataFormat::Int32:
        case DataFormat::UInt32: bytes_per_element = 4; break;
        default: bytes_per_element = 4; break;
    }

    const uint32_t output_tile_bytes = ublock_size_bytes;

    InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_base_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_base_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    for (uint32_t i = start_id; i < start_id + num_rows_per_core; ++i) {
        const uint32_t i0 = i / input_tile_offset;
        const uint32_t i1 = i % input_tile_offset;
        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            const uint32_t write_tile_id = get_tile_id(i0, i1, j, tiles_per_row, input_tile_offset);
            cb_wait_front(cb_out, ONE_TILE);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            if (is_output_dram) {
                noc_async_write_tile(write_tile_id, dram_output_addrg, l1_read_addr);
            } else {
                noc_async_write_tile(write_tile_id, l1_output_addrg, l1_read_addr);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, ONE_TILE);
        }
    }
}
