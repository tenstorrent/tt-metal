// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t end_id = start_id + num_tiles;

    const uint32_t tile_size_bytes = get_tile_size(in_cb_id);
    constexpr auto input_args = TensorAccessorArgs<1>();
    const auto input_addrg = TensorAccessor(input_args, input_addr, tile_size_bytes);

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(in_cb_id, 1);
        uint32_t in_cb_write_ptr = get_write_ptr(in_cb_id);
        noc_async_read_tile(i, input_addrg, in_cb_write_ptr);
        noc_async_read_barrier();
        cb_push_back(in_cb_id, 1);
    }
}
