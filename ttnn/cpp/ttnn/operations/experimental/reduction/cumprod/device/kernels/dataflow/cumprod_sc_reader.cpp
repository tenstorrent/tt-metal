// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../cumprod_common.hpp"

#include "debug/dprint.h"

namespace {
union Scaler {
    float f;
    uint32_t u;
};

template <bool is_input_dram>
FORCE_INLINE void read_tile_into_cb(
    const uint32_t& batch,
    const uint32_t& channel,
    const uint32_t& ht,
    const uint32_t& wt,
    const CumprodCompileTimeArgs& args,
    const InterleavedAddrGenFast<is_input_dram>& addr_gtor) {
    cb_reserve_back(args.cb_input, ONE_TILE);
    const uint32_t l1_write_addr{get_write_ptr(args.cb_input)};
    const uint32_t selected_tile{select_tile(batch, channel, ht, wt, args)};
    noc_async_read_tile(selected_tile, addr_gtor, l1_write_addr);
    noc_async_read_barrier();

    const uint32_t l1_read_addr{get_read_ptr(args.cb_input)};
    auto reader{reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr)};
    // DPRINT << "b/c/ht/wt: " << batch << "/" << channel << "/" << ht << "/" << wt << ", tile (uint16_t): " <<
    // reader[0] << " and " << reader[1] << ENDL();

    // auto reader2{reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr)};
    // DPRINT << "b/c/ht/wt: " << batch << "/" << channel << "/" << ht << "/" << wt << ", tile (uint32_t): " <<
    // reader2[0] << " and " << reader2[1] << ENDL();

    cb_push_back(args.cb_input, ONE_TILE);
    // DPRINT << "READ_TILE_22" << ENDL();
}

template <bool is_input_dram>
FORCE_INLINE void read_tiles_into_cb(
    const CumprodCompileTimeArgs& compile_time_args,
    const InterleavedAddrGenFast<is_input_dram>& addr_gtor,
    const DataFormat& input_data_format) {
    Scaler scaler;
    bool is_32{true};
    switch (static_cast<uint>(input_data_format) & 0x1F) {
        case static_cast<uint8_t>(DataFormat::Int32):
        case static_cast<uint8_t>(DataFormat::UInt32): scaler.u = 1; break;
        case static_cast<uint8_t>(DataFormat::Float16_b): is_32 = false;
        case static_cast<uint8_t>(DataFormat::Float32):
        default: scaler.f = 1.0f; break;
    }
    for (uint32_t b{0}; b < compile_time_args.batches; ++b) {
        for (uint32_t ht{0}; ht < compile_time_args.height_tiles; ++ht) {
            for (uint32_t wt{0}; wt < compile_time_args.width_tiles; ++wt) {
                for (uint32_t c{0}; c < compile_time_args.channels; ++c) {
                    read_tile_into_cb(b, c, ht, wt, compile_time_args, addr_gtor);
                }
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr auto compile_time_args{get_compile_time_args()};

    const uint32_t input_tile_byte_count{get_tile_size(compile_time_args.cb_input)};
    const DataFormat input_data_format{get_dataformat(compile_time_args.cb_input)};

    const uint32_t src_addr{get_arg_val<uint32_t>(0)};

    const InterleavedAddrGenFast<compile_time_args.is_input_dram> addr_gtor{
        .bank_base_address = src_addr, .page_size = input_tile_byte_count, .data_format = input_data_format};

    read_tiles_into_cb(compile_time_args, addr_gtor, input_data_format);
}
