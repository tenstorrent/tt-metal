// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;

    const InterleavedAddrGenFast<input_is_dram> s = {
        .bank_base_address = get_arg_val<uint32_t>(0),
        .page_size = get_tile_size(src_cb_id),
        .data_format = get_dataformat(src_cb_id),
    };

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(src_cb_id, 1);
        noc_async_read_tile(i, s, get_write_ptr(src_cb_id));
        noc_async_read_barrier();
        cb_push_back(src_cb_id, 1);
    }
}
