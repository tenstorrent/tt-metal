// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../cumprod_common.hpp"

namespace {

constexpr union {
    float f;
    int32_t u;
} caster{.f = 1.0f};

}  // namespace

void kernel_main() {
    constexpr bool input_dram = get_compile_time_arg_val(0) == 1;

    uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    uint32_t input_tile_offset = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    // This is the offset of all dimensions below the cumulation axis
    uint32_t low_rank_offset = get_arg_val<uint32_t>(5);
    // This is the offset of all dimensions above the cumulation axis (HtWt for last two axes)
    uint32_t high_rank_offset = get_arg_val<uint32_t>(6);

    cb_reserve_back(cb_one, ONE_TILE);
    uint32_t data_one_addr = get_write_ptr(cb_one);

    const int32_t ACC_START_VALUE_F32{caster.u};
    constexpr int32_t ACC_START_VALUE_F16{0x3F80};
    // TODO(jbbieniekTT): the below ones will work only if applied LLK is preconfigured appropriately for those (issue
    // #21108)
    constexpr int32_t ACC_START_VALUE_I32{0x1};
    constexpr int32_t ACC_START_VALUE_I16{0x1};
    constexpr int32_t ACC_START_VALUE_I8{0x1};

    const auto& input_data_format = get_dataformat(cb_in);

    uint32_t ublock_size_bytes = get_tile_size(cb_in);

    const uint32_t input_tile_bytes = ublock_size_bytes;
    const uint32_t output_tile_bytes = ublock_size_bytes;
    uint32_t scaler = 0;

    uint32_t bytes_per_element = 4;
    switch (input_data_format) {
        case DataFormat::Float32:
            scaler = ACC_START_VALUE_F32;
            bytes_per_element = 4;
            break;
        case DataFormat::Float16_b:
        case DataFormat::Float16:
            scaler = (ACC_START_VALUE_F16 << 16) | ACC_START_VALUE_F16;
            bytes_per_element = 2;
            break;
        case DataFormat::UInt8:
            scaler = (ACC_START_VALUE_I8 << 24) | (ACC_START_VALUE_I8 << 16) | (ACC_START_VALUE_I8 << 8) |
                     (ACC_START_VALUE_I8);
            bytes_per_element = 1;
            break;
        case DataFormat::UInt16:
            scaler = (ACC_START_VALUE_I16 << 16) | ACC_START_VALUE_I16;
            bytes_per_element = 2;
            break;
        case DataFormat::Int32:
        case DataFormat::UInt32:
            scaler = ACC_START_VALUE_I32;
            bytes_per_element = 4;
            break;
        default:
            scaler = 1;
            bytes_per_element = 4;
            break;
    }

    // TODO(jbbieniekTT): issue #21108
    int32_t* data_one = (int32_t*)data_one_addr;
    for (uint32_t i = 0; i < ublock_size_bytes / sizeof(data_one); i++) {
        data_one[i] = scaler;
    }

    InterleavedAddrGenFast<input_dram> input_addrg = {
        .bank_base_address = input_base_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    cb_push_back(cb_one, ONE_TILE);

    for (uint32_t i = start_id; i < start_id + num_rows_per_core; ++i) {
        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            const uint32_t read_tile_id{
                get_tile_id(low_rank_offset, high_rank_offset, j, tiles_per_row, input_tile_offset)};
            cb_reserve_back(cb_in, ONE_TILE);
            uint32_t l1_write_addr{get_write_ptr(cb_in)};
            noc_async_read_tile(read_tile_id, input_addrg, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in, ONE_TILE);
        }
        ++high_rank_offset;
        if (high_rank_offset >= input_tile_offset) {
            high_rank_offset = 0;
            ++low_rank_offset;
        }
    }
}
