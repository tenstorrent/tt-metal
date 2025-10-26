// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const auto s = TensorAccessor(dst_args, output_buffer_address, get_tile_size(dst_cb_id));

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(dst_cb_id, 1);
        uint32_t dst_cb_read_addr = get_read_ptr(dst_cb_id);
        noc_async_write_tile(i, s, dst_cb_read_addr);
        noc_async_write_barrier();
        cb_pop_front(dst_cb_id, 1);
    }
}
