// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../accumulation_common.hpp"

namespace {

constexpr union {
    float f;
    int32_t u;
} caster{.f = 1.0f};

}  // namespace

void kernel_main() {
    uint32_t input_base_addr = get_arg_val<uint32_t>(0);

    constexpr bool input_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    const uint32_t input_tile_offset = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);
    // This is the offset of all dimensions below the accumulation axis
    uint32_t low_rank_offset = get_arg_val<uint32_t>(5);
    // This is the offset of all dimensions above the accumulation axis (HtWt for last two axes)
    uint32_t high_rank_offset = get_arg_val<uint32_t>(6);
    // backward flag (from n-1 to 0)
    const uint32_t flip = get_arg_val<uint32_t>(7);
    // type of accumulation
    const AccumulationOp accumulation_op = static_cast<AccumulationOp>(get_arg_val<uint32_t>(8));

    cb_reserve_back(cb_start, ONE_TILE);
    uint32_t data_start_addr = get_write_ptr(cb_start);

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
    uint32_t scaler = 0;

    if (accumulation_op == AccumulationOp::CUMPROD) {
        switch (input_data_format) {
            case DataFormat::Float32: scaler = ACC_START_VALUE_F32; break;
            case DataFormat::Float16_b:
            case DataFormat::Float16: scaler = (ACC_START_VALUE_F16 << 16) | ACC_START_VALUE_F16; break;
            case DataFormat::UInt8:
                scaler = (ACC_START_VALUE_I8 << 24) | (ACC_START_VALUE_I8 << 16) | (ACC_START_VALUE_I8 << 8) |
                         (ACC_START_VALUE_I8);
                break;
            case DataFormat::UInt16: scaler = (ACC_START_VALUE_I16 << 16) | ACC_START_VALUE_I16; break;
            case DataFormat::Int32:
            case DataFormat::UInt32: scaler = ACC_START_VALUE_I32; break;
            default: scaler = 1; break;
        }
    } else if (accumulation_op == AccumulationOp::CUMSUM) {
        scaler = 0;
    }

    // TODO(jbbieniekTT): issue #21108
    volatile tt_l1_ptr int32_t* data_start = reinterpret_cast<volatile tt_l1_ptr int32_t*>(data_start_addr);
    for (uint32_t i = 0; i < ublock_size_bytes / sizeof(data_start); i++) {
        data_start[i] = scaler;
    }

    InterleavedAddrGenFast<input_dram> input_addrg = {
        .bank_base_address = input_base_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    cb_push_back(cb_start, ONE_TILE);

    for (uint32_t i = start_id; i < start_id + num_rows_per_core; ++i) {
        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            const uint32_t tile_j = flip ? (tiles_per_row - j - 1) : j;
            const uint32_t read_tile_id{
                get_tile_id(low_rank_offset, high_rank_offset, tile_j, tiles_per_row, input_tile_offset)};
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
