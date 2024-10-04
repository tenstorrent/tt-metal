// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t value = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_value = tt::CB::c_intermed0;
    constexpr uint32_t onetile = 1;

    const uint32_t src_tile_bytes = get_tile_size(cb_value);
    const DataFormat src_data_format = get_dataformat(cb_value);

    const InterleavedAddrGenFast<src0_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = src_tile_bytes,
        .data_format = src_data_format
    };

    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_value, onetile);
        const auto cb0_l1_addr = get_write_ptr(cb_value);
        noc_async_read_tile(i, s, cb0_l1_addr, 0);
        noc_async_read_barrier();

        cb_push_back(cb_value, 1);
    }


}
