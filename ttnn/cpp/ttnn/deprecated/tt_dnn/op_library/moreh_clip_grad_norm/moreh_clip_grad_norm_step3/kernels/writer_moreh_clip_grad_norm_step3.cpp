// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const bool output_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_output = 16;

    constexpr uint32_t onetile = 1;

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_data_format = get_dataformat(cb_id_output);

    const InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};
    const InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    // output
    const auto output_l1_read_addr = get_read_ptr(cb_id_output);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_wait_front(cb_id_output, onetile);
        if (output_is_dram) {
            noc_async_write_tile(tile_idx, dram_output_addrg, output_l1_read_addr);
        } else {
            noc_async_write_tile(tile_idx, l1_output_addrg, output_l1_read_addr);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_output, onetile);
    }

}  // void kernel_main()
