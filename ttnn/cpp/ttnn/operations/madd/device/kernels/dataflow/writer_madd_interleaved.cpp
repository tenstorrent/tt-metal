// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t num_pages_per_tile_row = get_compile_time_arg_val(2);

    constexpr auto dst_args = TensorAccessorArgs<3>();
    const auto s0 = TensorAccessor(dst_args, dst_addr, page_size);

    const uint32_t end_id = start_page_id + num_pages;

    // reader copied the data from DRAM to CB buffer.
    for (uint32_t i = start_page_id; i < end_id; i += num_pages_per_tile_row) {
        cb_wait_front(cb_id_out, num_pages_per_tile_row);
        uint64_t base_l1_read_addr = get_read_ptr(cb_id_out);

        uint64_t dst_noc_addr = s0.get_noc_addr(i);

        // write entire tile row at once
        // should I go tile by tile?
        // TODO: find suitable tile count to write at once based on perf.
        noc_async_write(base_l1_read_addr, dst_noc_addr, page_size * num_pages_per_tile_row);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out, num_pages_per_tile_row);
    }
}
