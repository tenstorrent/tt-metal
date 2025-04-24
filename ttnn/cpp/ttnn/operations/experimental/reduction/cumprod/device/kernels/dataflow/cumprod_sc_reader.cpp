// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

namespace {

FORCE_INLINE unsigned get_tile_id(
    uint32_t i0, uint32_t i1, uint32_t j, uint32_t tiles_per_row, uint32_t PLo, uint32_t PHi, uint32_t HtWt) {
    uint32_t base_tileid = i0 * (tiles_per_row * PHi * HtWt) + i1;
    uint32_t tileid = base_tileid + j * PHi * HtWt;
    return tileid;
}

constexpr union {
    float f;
    int32_t u;
} caster{.f = 1.0f};

}  // namespace

void kernel_main() {
    uint32_t input_dram_base_addr = get_arg_val<uint32_t>(0);  // input base addr (DRAM)
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);  // number of tiles in a row / along axis
    uint32_t PHi = get_arg_val<uint32_t>(3);
    uint32_t PLo = get_arg_val<uint32_t>(4);
    uint32_t HtWt = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = tt::CBIndex::c_0;
    constexpr uint32_t cb_one = tt::CBIndex::c_16;

    cb_reserve_back(cb_one, 1);
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
    InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_dram_base_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    uint32_t scaler{0};

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
        default:  // ?
            scaler = 1;
            bytes_per_element = 4;
            break;
    }

    const uint32_t tile_card = ublock_size_bytes / bytes_per_element;

    int32_t* data_one{(int32_t*)data_one_addr};
    for (uint32_t i = 0; i < ublock_size_bytes / 4; i++) {
        data_one[i] = scaler;
    }

    cb_push_back(cb_one, 1);

    for (unsigned i0 = 0; i0 < PLo; i0++) {
        for (unsigned i1 = 0; i1 < PHi * HtWt; i1++) {
            for (unsigned j = 0; j < tiles_per_row; j++) {
                uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);

                cb_reserve_back(cb_out, 1);

                uint32_t data_sram_addr = get_write_ptr(cb_out);
                noc_async_read_tile(tileid, dram_input_addrg, data_sram_addr);
                noc_async_read_barrier();

                cb_push_back(cb_out, 1);
            }
        }
    }
}
