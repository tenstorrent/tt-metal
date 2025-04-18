// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "../cumprod_common.hpp"

#include "debug/dprint.h"

#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

namespace {
union Scaler {
    float f;
    uint32_t u;
};

FORCE_INLINE unsigned get_tile_id(
    uint32_t i0, uint32_t i1, uint32_t j, uint32_t tiles_per_row, uint32_t PLo, uint32_t PHi, uint32_t HtWt) {
    uint32_t base_tileid = i0 * (tiles_per_row * PHi * HtWt) + i1;
    uint32_t tileid = base_tileid + j * PHi * HtWt;
    return tileid;
}

// template <bool is_input_dram>
// FORCE_INLINE void read_tile_into_cb(
//     const uint32_t& batch,
//     const uint32_t& channel,
//     const uint32_t& ht,
//     const uint32_t& wt,
//     const CumprodCompileTimeArgs& args,
//     const InterleavedAddrGenFast<is_input_dram>& addr_gtor) {
//     cb_reserve_back(args.cb_input, ONE_TILE);
//     const uint32_t l1_write_addr{get_write_ptr(args.cb_input)};
//     const uint32_t selected_tile{select_tile(batch, channel, ht, wt, args)};
//     noc_async_read_tile(selected_tile, addr_gtor, l1_write_addr);
//     noc_async_read_barrier();

//     const uint32_t l1_read_addr{get_read_ptr(args.cb_input)};
//     auto reader{reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr)};
//     // DPRINT << "b/c/ht/wt: " << batch << "/" << channel << "/" << ht << "/" << wt << ", tile (uint16_t): " <<
//     // reader[0] << " and " << reader[1] << ENDL();

//     // auto reader2{reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr)};
//     // DPRINT << "b/c/ht/wt: " << batch << "/" << channel << "/" << ht << "/" << wt << ", tile (uint32_t): " <<
//     // reader2[0] << " and " << reader2[1] << ENDL();

//     cb_push_back(args.cb_input, ONE_TILE);
//     // DPRINT << "READ_TILE_22" << ENDL();
// }

// template <bool is_input_dram>
// FORCE_INLINE void read_tiles_into_cb(
//     const CumprodCompileTimeArgs& compile_time_args,
//     const InterleavedAddrGenFast<is_input_dram>& addr_gtor,
//     const DataFormat& input_data_format) {
//     Scaler scaler;
//     bool is_32{true};
//     switch (static_cast<uint>(input_data_format) & 0x1F) {
//         case static_cast<uint8_t>(DataFormat::Int32):
//         case static_cast<uint8_t>(DataFormat::UInt32): scaler.u = 1; break;
//         case static_cast<uint8_t>(DataFormat::Float16_b): is_32 = false;
//         case static_cast<uint8_t>(DataFormat::Float32):
//         default: scaler.f = 1.0f; break;
//     }
//     for (uint32_t b{0}; b < compile_time_args.batches; ++b) {
//         for (uint32_t ht{0}; ht < compile_time_args.height_tiles; ++ht) {
//             for (uint32_t wt{0}; wt < compile_time_args.width_tiles; ++wt) {
//                 for (uint32_t c{0}; c < compile_time_args.channels; ++c) {
//                     read_tile_into_cb(b, c, ht, wt, compile_time_args, addr_gtor);
//                 }
//             }
//         }
//     }
// }
}  // namespace

void kernel_main() {
    // constexpr auto compile_time_args{get_compile_time_args()};

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

    // union {
    //     float f;
    //     uint32_t u32;
    //     uint16_t u16;
    // } scaler;

    // const DataFormat data_format = get_dataformat(cb_one);
    // switch ((uint)data_format & 0x1F) {
    //     case ((uint8_t)DataFormat::Int32):
    //     case ((uint8_t)DataFormat::UInt32): scaler.u32 = 1; break;
    //     case ((uint8_t)DataFormat::Float32): scaler.f = 1.0f; break;
    //     case ((uint8_t)DataFormat::Float16_b): scaler.u16 = 0x380; break;
    //     case ((uint8_t)DataFormat::UInt8): scaler.u32 = 1; break;
    //     default: scaler.f = 1.0f; break;
    // }

    // fill_cb_with_value(cb_one, scaler.u32);

    union {
        float f;
        int32_t u;
    } caster;

    caster.f = 1.0f;

    int32_t ACC_START_VALUE_F32{caster.u};
    constexpr int32_t ACC_START_VALUE_F16{0x3F80};  // 1.0f for bfloat16
    constexpr int32_t ACC_START_VALUE_I32{0x70007000};
    constexpr int32_t ACC_START_VALUE_I16{1};
    constexpr int32_t ACC_START_VALUE_I8{8};

    const auto& input_data_format = get_dataformat(cb_out);

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_out);
    uint32_t l1_addr_out = get_write_ptr(cb_out);

    const uint32_t input_tile_bytes = ublock_size_bytes;
    const uint32_t output_tile_bytes = ublock_size_bytes;
    InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_dram_base_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    // union {
    //     int32_t i32;
    //     int16_t i16;
    //     int8_t i8;
    // } scaler;
    // scaler.i32 = 0;
    uint32_t scaler{0};

    uint32_t bytes_per_element = 4;
    switch (input_data_format) {
        case DataFormat::Float32:
            scaler = ACC_START_VALUE_F32;
            DPRINT << "F32" << ENDL();
            bytes_per_element = 4;
            break;
        case DataFormat::Float16_b:
        case DataFormat::Float16:
            scaler = (ACC_START_VALUE_F16 << 16) | ACC_START_VALUE_F16;
            DPRINT << "F16" << ENDL();
            bytes_per_element = 2;
            break;
        case DataFormat::UInt8:
            scaler = (ACC_START_VALUE_I8 << 24) | (ACC_START_VALUE_I8 << 16) | (ACC_START_VALUE_I8 << 8) |
                     (ACC_START_VALUE_I8);
            DPRINT << "U8" << ENDL();
            bytes_per_element = 1;
            break;
        case DataFormat::UInt16:
            scaler = (ACC_START_VALUE_I16 << 16) | ACC_START_VALUE_I16;
            DPRINT << "U16" << ENDL();
            bytes_per_element = 2;
            break;
        case DataFormat::Int32:
        case DataFormat::UInt32:
            scaler = ACC_START_VALUE_I32;
            DPRINT << "U32" << ENDL();
            bytes_per_element = 4;
            break;
        default:  // ?
            scaler = 1;
            bytes_per_element = 4;
            break;
    }

    DPRINT << "bytes_per_element: " << bytes_per_element << ", ublock_size_bytes: " << ublock_size_bytes << ENDL();

    const uint32_t tile_card = ublock_size_bytes / bytes_per_element;

    int32_t* data_one{(int32_t*)data_one_addr};
    for (uint32_t i = 0; i < ublock_size_bytes / 4; i++) {
        data_one[i] = scaler;
    }

    // switch (bytes_per_element) {
    //     case 1: {
    //         int8_t* data_one{(int8_t*) data_one_addr};
    //         for (uint32_t i = 0; i < ublock_size_bytes / bytes_per_element; i++) {
    //             data_one[i] = scaler.i8;
    //         }
    //     }

    //     case 2: {
    //         int16_t* data_one{(int16_t*) data_one_addr};
    //         for (uint32_t i = 0; i < ublock_size_bytes / bytes_per_element; i++) {
    //             data_one[i] = scaler.i16;
    //         }
    //     }

    //     case 4: {
    //         int32_t* data_one{(int32_t*) data_one_addr};
    //         for (uint32_t i = 0; i < ublock_size_bytes / bytes_per_element; i++) {
    //             data_one[i] = scaler.i32;
    //         }
    //     }

    //     // TODO(jbbieniekTT): ?
    //     default: {
    //         // int16_t* data_one{(int16_t*) data_one_addr};
    //         // for (uint32_t i = 0; i < ublock_size_bytes / bytes_per_element; i++) {
    //         //     data_one[i] = scaler.i16;
    //         // }
    //     }
    // }
    cb_push_back(cb_one, 1);

    DPRINT << "[Cumprod Reader] #tiles/row = " << tiles_per_row << ", tile size = " << ublock_size_bytes
           << ", Bytes/Element = " << bytes_per_element << ", tile  card = " << tile_card << ", num rows = " << num_rows
           << ", PHi = " << PHi << ", PLo = " << PLo << ", HtWt = " << HtWt << ENDL();

    for (unsigned i0 = 0; i0 < PLo; i0++) {
        for (unsigned i1 = 0; i1 < PHi * HtWt; i1++) {
            for (unsigned j = 0; j < tiles_per_row; j++) {
                uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);

                cb_reserve_back(cb_out, 1);

                // DPRINT << "[READER] 1" << ENDL();

                // Read tile
                uint32_t data_sram_addr = get_write_ptr(cb_out);
                noc_async_read_tile(tileid, dram_input_addrg, data_sram_addr);
                noc_async_read_barrier();

                // Write tile
                cb_push_back(cb_out, 1);
                // DPRINT << "[READER] 2" << ENDL();
            }
        }
    }

    // const uint32_t input_tile_byte_count{get_tile_size(compile_time_args.cb_input)};
    // const DataFormat input_data_format{get_dataformat(compile_time_args.cb_input)};

    // const uint32_t src_addr{get_arg_val<uint32_t>(0)};

    // const InterleavedAddrGenFast<compile_time_args.is_input_dram> addr_gtor{
    //     .bank_base_address = src_addr, .page_size = input_tile_byte_count, .data_format = input_data_format};

    // read_tiles_into_cb(compile_time_args, addr_gtor, input_data_format);
}
