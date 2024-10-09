// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t output_addr = get_arg_val<uint32_t>(0);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(1);
    std::uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_value = tt::CB::c_intermed0;
    const uint32_t cb_page_size = get_tile_size(cb_value);
    const auto cb_data_format = get_dataformat(cb_value);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = output_addr, .page_size = cb_page_size, .data_format = cb_data_format};

    cb_wait_front(cb_value, 1);

    uint32_t write_addr = get_read_ptr(cb_value);
    auto ptr = reinterpret_cast<uint32_t *>(write_addr);
    uint32_t end_id = start_id + num_tiles;
    for (std::uint32_t i = start_id; i < end_id; i++) {
        const auto cb_value_addr = get_read_ptr(cb_value);
        noc_async_write_tile(i, s, cb_value_addr);
        noc_async_write_barrier();
    }
    cb_pop_front(cb_value, 1);
}
