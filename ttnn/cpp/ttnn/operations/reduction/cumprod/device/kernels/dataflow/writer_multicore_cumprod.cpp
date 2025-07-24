// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../cumprod_common.hpp"

void kernel_main() {
    constexpr bool output_dram = get_compile_time_arg_val(0) == 1;

    uint32_t output_base_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    uint32_t input_tile_offset = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    // This is the offset of all dimensions below the cumulation axis
    uint32_t low_rank_offset = get_arg_val<uint32_t>(5);
    // This is the offset of all dimensions above the cumulation axis (HtWt for last two axes)
    uint32_t high_rank_offset = get_arg_val<uint32_t>(6);

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

    InterleavedAddrGenFast<output_dram> output_addrg = {
        .bank_base_address = output_base_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    for (uint32_t i = start_id; i < start_id + num_rows_per_core; ++i) {
        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            const uint32_t write_tile_id =
                get_tile_id(low_rank_offset, high_rank_offset, j, tiles_per_row, input_tile_offset);
            cb_wait_front(cb_out, ONE_TILE);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            noc_async_write_tile(write_tile_id, output_addrg, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out, ONE_TILE);
        }
        ++high_rank_offset;
        if (high_rank_offset >= input_tile_offset) {
            high_rank_offset = 0;
            ++low_rank_offset;
        }
    }
}
