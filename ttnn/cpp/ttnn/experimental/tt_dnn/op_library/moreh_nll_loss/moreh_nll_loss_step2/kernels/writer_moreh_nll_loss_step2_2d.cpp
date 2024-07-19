// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    auto output_addr = get_arg_val<uint32_t>(0);
    auto num_tiles_per_core = get_arg_val<uint32_t>(1);
    auto start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output = tt::CB::c_out0;

    const uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto output_data_format = get_dataformat(cb_output);

    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    constexpr uint32_t onetile = 1;
    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_output, onetile);
        uint32_t output_l1_write_addr = get_read_ptr(cb_output);
        noc_async_write_tile(i, output_addrg, output_l1_write_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output, onetile);
    }
}
