// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "../cumprod_common.hpp"

#include "dataflow_api.h"

template <bool is_input_dram>
void send_tile_from_cb(
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

    cb_pop_front(args.cb_output, ONE_TILE);
}

namespace NAMESPACE {
void MAIN {
    constexpr auto compile_time_args{get_compile_time_args()};

    const uint32_t output_tile_byte_count{get_tile_size(static_cast<const DataFormat>(compile_time_args.cb_output))};
    const DataFormat output_data_format{get_dataformat(compile_time_args.cb_output)};
    const InterleavedAddrGenFast<compile_time_args.is_output_dram> addr_gtor{
        .bank_base_address = compile_time_args.dst_addr,
        .page_size = output_tile_byte_count,
        .data_format = output_data_format};

    using namespace std::placeholders;
    for_each_tile_grouped_by_channels(compile_time_args, std::bind(&send_tile_from_cb, _1, _2, _3, _4, _5, addr_gtor));
}
}  // namespace NAMESPACE
