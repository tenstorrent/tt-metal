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
    uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    uint32_t input_tile_offset = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    uint32_t is_input_dram = get_arg_val<uint32_t>(5);

    cb_reserve_back(cb_one, ONE_TILE);
    uint32_t data_one_addr = get_write_ptr(cb_one);

    const int32_t ACC_START_VALUE_F32{caster.u};
    constexpr int32_t ACC_START_VALUE_F16{0x3F80};
    // TODO(jbbieniekTT): the below ones will work only if applied LLK is preconfigured appropriately for those.
    constexpr int32_t ACC_START_VALUE_I32{0x1};
    constexpr int32_t ACC_START_VALUE_I16{0x1};
    constexpr int32_t ACC_START_VALUE_I8{0x1};

    const auto& input_data_format = get_dataformat(cb_out);

    uint32_t ublock_size_bytes = get_tile_size(cb_out);
    uint32_t l1_addr_out = get_write_ptr(cb_out);

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

    // TODO(jbbieniekTTT): issue #
    int32_t* data_one = (int32_t*)data_one_addr;
    for (uint32_t i = 0; i < ublock_size_bytes / sizeof(data_one); i++) {
        data_one[i] = scaler;
    }

    InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_base_addr, .page_size = output_tile_bytes, .data_format = input_data_format};

    InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_base_addr, .page_size = output_tile_bytes, .data_format = input_data_format};

    cb_push_back(cb_one, ONE_TILE);

    for (uint32_t i = start_id; i < start_id + num_rows_per_core; ++i) {
        const uint32_t i0 = i / input_tile_offset;
        const uint32_t i1 = i % input_tile_offset;
        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            const uint32_t read_tile_id{get_tile_id(i0, i1, j, tiles_per_row, input_tile_offset)};
            cb_reserve_back(cb_in, ONE_TILE);
            uint32_t l1_write_addr_in0{get_write_ptr(cb_in)};
            if (is_input_dram) {
                noc_async_read_tile(read_tile_id, dram_input_addrg, l1_write_addr_in0);
            } else {
                noc_async_read_tile(read_tile_id, l1_input_addrg, l1_write_addr_in0);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, ONE_TILE);
        }
    }
}
