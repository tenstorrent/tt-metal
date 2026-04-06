// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

#include "../accumulation_common.hpp"

void kernel_main() {
    uint32_t input_base_addr = get_arg_val<uint32_t>(0);

    constexpr auto input_addrg_args = TensorAccessorArgs<0>();
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

    experimental::Noc noc;
    experimental::CircularBuffer cb_start_obj(cb_start);
    experimental::CircularBuffer cb_in_obj(cb_in);

    cb_start_obj.reserve_back(ONE_TILE);
    uint32_t data_start_addr = cb_start_obj.get_write_ptr();

    const float one_f = 1.0f;
    int32_t ACC_START_VALUE_F32 = 0;
    std::memcpy(&ACC_START_VALUE_F32, &one_f, sizeof(int32_t));
    constexpr int32_t ACC_START_VALUE_F16{0x3F80};
    // TODO(jbbieniekTT): the below ones will work only if applied LLK is preconfigured appropriately for those (issue
    // #21108)
    constexpr int32_t ACC_START_VALUE_I32{0x1};
    constexpr int32_t ACC_START_VALUE_I16{0x1};
    constexpr int32_t ACC_START_VALUE_I8{0x1};

    uint32_t ublock_size_bytes = get_tile_size(cb_in);

    const uint32_t input_tile_bytes = ublock_size_bytes;
    uint32_t scaler = 0;

    if (accumulation_op == AccumulationOp::CUMPROD) {
        const auto& input_data_format = get_dataformat(cb_in);
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
    experimental::CoreLocalMem<volatile int32_t> data_start(data_start_addr);
    for (uint32_t i = 0; i < ublock_size_bytes / sizeof(int32_t); i++) {
        data_start[i] = scaler;
    }

    const auto input_addrg = TensorAccessor(input_addrg_args, input_base_addr, input_tile_bytes);

    cb_start_obj.push_back(ONE_TILE);

    for (uint32_t i = start_id; i < start_id + num_rows_per_core; ++i) {
        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            const uint32_t tile_j = flip ? (tiles_per_row - j - 1) : j;
            const uint32_t read_tile_id{
                get_tile_id(low_rank_offset, high_rank_offset, tile_j, tiles_per_row, input_tile_offset)};
            cb_in_obj.reserve_back(ONE_TILE);
            noc.async_read(input_addrg, cb_in_obj, input_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in_obj.push_back(ONE_TILE);
        }
        ++high_rank_offset;
        if (high_rank_offset >= input_tile_offset) {
            high_rank_offset = 0;
            ++low_rank_offset;
        }
    }
}
