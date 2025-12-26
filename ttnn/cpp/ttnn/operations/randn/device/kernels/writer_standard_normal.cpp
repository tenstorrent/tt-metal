// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "dataflow_api.h"

using namespace tt;

void kernel_main() {
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t num_pairs = num_tiles >> 1;

    const uint32_t is_odd = num_tiles & 1;

    uint32_t dst_tile_bytes = get_tile_size(dst_cb_id);

    const auto output_addrg = TensorAccessor(dst_args, dst_addr, get_tile_size(dst_cb_id));

    for (uint32_t p = 0; p < num_pairs; p++) {
        uint32_t i = start_id + (p << 1);
        cb_wait_front(dst_cb_id, 2);

        uint32_t dst_cb_read_base = get_read_ptr(dst_cb_id);
        uint32_t dst_cb_read0_ptr = dst_cb_read_base;
        uint32_t dst_cb_read1_ptr = dst_cb_read_base + dst_tile_bytes;

        noc_async_write_tile(i, output_addrg, dst_cb_read0_ptr);
        noc_async_write_tile(i + 1, output_addrg, dst_cb_read1_ptr);
        noc_async_write_barrier();
        cb_pop_front(dst_cb_id, 2);
    }

    if (is_odd) {
        uint32_t i = start_id + (num_pairs << 1);
        cb_wait_front(dst_cb_id, 1);

        uint32_t dst_cb_read0_ptr = get_read_ptr(dst_cb_id);

        noc_async_write_tile(i, output_addrg, dst_cb_read0_ptr);
        noc_async_write_barrier();
        cb_pop_front(dst_cb_id, 1);
    }
}
