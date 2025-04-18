// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "../cumprod_common.hpp"

#include "debug/dprint.h"

// namespace {
// template <bool is_input_dram>
// FORCE_INLINE void send_tile_from_cb(
//     const uint32_t& batch,
//     const uint32_t& channel,
//     const uint32_t& ht,
//     const uint32_t& wt,
//     const CumprodCompileTimeArgs& args,
//     const InterleavedAddrGenFast<is_input_dram>& addr_gtor) {
//     cb_wait_front(args.cb_output, ONE_TILE);
//     const uint32_t l1_read_addr{get_read_ptr(args.cb_output)};
//     const uint32_t selected_tile{select_tile(batch, channel, ht, wt, args)};
//     noc_async_write_tile(selected_tile, addr_gtor, l1_read_addr);
//     noc_async_write_barrier();

//     const uint32_t l1_read_addr_{get_read_ptr(args.cb_output)};
//     auto reader{reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr_)};
//     // DPRINT << "b/c/ht/wt: " << batch << "/" << channel << "/" << ht << "/" << wt << ", tile (uint16_t): " <<
//     // reader[0] << " and " << reader[1] << ENDL();

//     cb_pop_front(args.cb_output, ONE_TILE);
// }

// template <bool is_input_dram>
// FORCE_INLINE void send_tiles_from_cb(
//     const CumprodCompileTimeArgs& compile_time_args, const InterleavedAddrGenFast<is_input_dram> addr_gtor) {
//     for (uint32_t b{0}; b < compile_time_args.batches; ++b) {
//         for (uint32_t ht{0}; ht < compile_time_args.height_tiles; ++ht) {
//             for (uint32_t wt{0}; wt < compile_time_args.width_tiles; ++wt) {
//                 for (uint32_t c{0}; c < compile_time_args.channels; ++c) {
//                     send_tile_from_cb(b, c, ht, wt, compile_time_args, addr_gtor);
//                 }
//             }
//         }
//     }
// }
// }  // namespace

// void kernel_main() {
//     constexpr auto compile_time_args{get_compile_time_args()};

//     const uint32_t output_tile_byte_count{get_tile_size(compile_time_args.cb_output)};
//     const DataFormat output_data_format{get_dataformat(compile_time_args.cb_output)};

//     const uint32_t dst_addr{get_arg_val<uint32_t>(1)};

//     const InterleavedAddrGenFast<compile_time_args.is_output_dram> addr_gtor{
//         .bank_base_address = dst_addr, .page_size = output_tile_byte_count, .data_format = output_data_format};

//     send_tiles_from_cb(compile_time_args, addr_gtor);
// }

FORCE_INLINE unsigned get_tile_id(
    uint32_t i0, uint32_t i1, uint32_t j, uint32_t tiles_per_row, uint32_t PLo, uint32_t PHi, uint32_t HtWt) {
    uint32_t base_tileid = i0 * (tiles_per_row * PHi * HtWt) + i1;
    uint32_t tileid = base_tileid + j * PHi * HtWt;
    return tileid;
}

void kernel_main() {
    uint32_t output_dram_base_addr = get_arg_val<uint32_t>(0);  // output base addr (DRAM)
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);  // number of tiles in a row / along axis
    uint32_t PHi = get_arg_val<uint32_t>(3);
    uint32_t PLo = get_arg_val<uint32_t>(4);
    uint32_t HtWt = get_arg_val<uint32_t>(5);

    // DPRINT <<

    constexpr uint32_t cb_in = tt::CBIndex::c_1;

    // single tile ublock
    uint32_t ublock_size_bytes = get_tile_size(cb_in);
    uint32_t input_sram_addr = get_read_ptr(cb_in);

    const auto& input_dataformat = get_dataformat(cb_in);
    const auto& output_data_format = get_dataformat(cb_in);

    union {
        float f;
        int32_t u;
    } caster;

    caster.f = 1.0f;

    int32_t ACC_START_VALUE_F32{caster.u};
    constexpr int16_t ACC_START_VALUE_F16{0x380};  // 1.0f for bfloat16
    constexpr int32_t ACC_START_VALUE_I32{1};
    constexpr int16_t ACC_START_VALUE_I16{1};
    constexpr int8_t ACC_START_VALUE_I8{1};

    uint32_t bytes_per_element = 4;

    switch (input_dataformat) {
        case DataFormat::Float32: bytes_per_element = 4; break;
        case DataFormat::Float16_b:
        case DataFormat::Float16: bytes_per_element = 2; break;
        case DataFormat::UInt8: bytes_per_element = 1; break;
        case DataFormat::UInt16: bytes_per_element = 2; break;
        case DataFormat::Int32:
        case DataFormat::UInt32: bytes_per_element = 4; break;
        default:  // TODO(jbbieniekTT): ?
            bytes_per_element = 4;
            break;
    }

    uint32_t tile_card = ublock_size_bytes / bytes_per_element;
    const uint32_t output_tile_bytes = ublock_size_bytes;

    InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_dram_base_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    for (unsigned i0 = 0; i0 < PLo; i0++) {
        for (unsigned i1 = 0; i1 < PHi * HtWt; i1++) {
            for (unsigned j = 0; j < tiles_per_row; j++) {
                uint32_t tileid = get_tile_id(i0, i1, j, tiles_per_row, PLo, PHi, HtWt);

                cb_wait_front(cb_in, 1);

                // Write tile
                noc_async_write_tile(tileid, dram_output_addrg, input_sram_addr);
                noc_async_write_barrier();

                cb_pop_front(cb_in, 1);
            }
        }
    }
}
