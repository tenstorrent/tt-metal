// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../cumprod_common.hpp"

#include "debug/dprint.h"

namespace {
template <bool is_input_dram>
FORCE_INLINE void send_tile_from_cb(
    const uint32_t& batch,
    const uint32_t& channel,
    const uint32_t& ht,
    const uint32_t& wt,
    const CumprodCompileTimeArgs& args,
    const InterleavedAddrGenFast<is_input_dram>& addr_gtor) {
    cb_wait_front(args.cb_output, ONE_TILE);
    const uint32_t l1_read_addr{get_read_ptr(args.cb_output)};
    const uint32_t selected_tile{select_tile(batch, channel, ht, wt, args)};
    noc_async_write_tile(selected_tile, addr_gtor, l1_read_addr);
    noc_async_write_barrier();

    const uint32_t l1_read_addr_{get_read_ptr(args.cb_output)};
    auto reader{reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr_)};
    // DPRINT << "b/c/ht/wt: " << batch << "/" << channel << "/" << ht << "/" << wt << ", tile (uint16_t): " <<
    // reader[0] << " and " << reader[1] << ENDL();

    cb_pop_front(args.cb_output, ONE_TILE);
}

template <bool is_input_dram>
FORCE_INLINE void send_tiles_from_cb(
    const CumprodCompileTimeArgs& compile_time_args, const InterleavedAddrGenFast<is_input_dram> addr_gtor) {
    for (uint32_t b{0}; b < compile_time_args.batches; ++b) {
        for (uint32_t ht{0}; ht < compile_time_args.height_tiles; ++ht) {
            for (uint32_t wt{0}; wt < compile_time_args.width_tiles; ++wt) {
                for (uint32_t c{0}; c < compile_time_args.channels; ++c) {
                    send_tile_from_cb(b, c, ht, wt, compile_time_args, addr_gtor);
                }
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr auto compile_time_args{get_compile_time_args()};

    const uint32_t output_tile_byte_count{get_tile_size(compile_time_args.cb_output)};
    const DataFormat output_data_format{get_dataformat(compile_time_args.cb_output)};

    const uint32_t dst_addr{get_arg_val<uint32_t>(1)};

    const InterleavedAddrGenFast<compile_time_args.is_output_dram> addr_gtor{
        .bank_base_address = dst_addr, .page_size = output_tile_byte_count, .data_format = output_data_format};

    send_tiles_from_cb(compile_time_args, addr_gtor);
}
