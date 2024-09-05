// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

inline uint32_t get_write_tile_id(uint32_t tile_id, uint32_t dim, uint32_t CHtWt, uint32_t HtWt) {
    return (dim == 0) ? (tile_id) : (tile_id / HtWt * CHtWt) + (tile_id % HtWt);
}

void kernel_main() {
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto num_tiles_to_cumsum = get_arg_val<uint32_t>(1);
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(2);
    const auto input_tile_offset = get_arg_val<uint32_t>(3);
    const auto start_id = get_arg_val<uint32_t>(4);
    const auto output_is_dram = (get_arg_val<uint32_t>(5) == 1);
    const auto HtWt = get_arg_val<uint32_t>(6);
    const auto CHtWt = get_arg_val<uint32_t>(7);
    const auto dim = get_arg_val<uint32_t>(8);
    const auto flip = (get_arg_val<uint32_t>(9) == 1);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;

    uint32_t output_tile_bytes = get_tile_size(cb_id_out);
    const auto output_data_format = get_dataformat(cb_id_out);
    const InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};
    const InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    for (uint32_t i = start_id; i < start_id + num_output_tiles_per_core; i++) {
        auto write_tile_id = get_write_tile_id(i, dim, CHtWt, HtWt);
        auto offset = input_tile_offset;
        if (flip) {
            write_tile_id += (num_tiles_to_cumsum - 1) * input_tile_offset;
            offset *= -1;
        }

        for (uint32_t j = 0; j < num_tiles_to_cumsum; ++j) {
            cb_wait_front(cb_id_out, onetile);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            if (output_is_dram) {
                noc_async_write_tile(write_tile_id, dram_output_addrg, l1_read_addr);
            } else {
                noc_async_write_tile(write_tile_id, l1_output_addrg, l1_read_addr);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, onetile);
            write_tile_id += offset;
        }
    }
}
