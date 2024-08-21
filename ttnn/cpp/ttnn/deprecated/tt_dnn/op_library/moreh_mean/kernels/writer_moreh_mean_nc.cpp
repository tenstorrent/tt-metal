// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto num_tiles = get_arg_val<uint32_t>(1);
    const auto start_id = get_arg_val<uint32_t>(2);
    const auto output_is_dram = (get_arg_val<uint32_t>(3) == 1);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;

    uint32_t output_tile_bytes = get_tile_size(cb_id_out);
    const auto output_data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};
    const InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        uint32_t write_tile_id = i;
        cb_wait_front(cb_id_out, onetile);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        if (output_is_dram) {
            noc_async_write_tile(write_tile_id, dram_output_addrg, l1_read_addr);
        } else {
            noc_async_write_tile(write_tile_id, l1_output_addrg, l1_read_addr);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
}
