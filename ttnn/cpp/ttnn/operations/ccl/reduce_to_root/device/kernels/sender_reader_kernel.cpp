// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_tiles_l = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(2);

    const uint32_t src_addr_l = get_arg_val<uint32_t>(0);
    const uint32_t src_addr_s = get_arg_val<uint32_t>(1);
    const uint32_t src_addr_m = get_arg_val<uint32_t>(2);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(4);
    constexpr uint32_t onetile = 1;

    cb_reserve_back(packet_cb_id, 1);
    uint32_t l1_write_addr = get_write_ptr(packet_cb_id);
    uint64_t read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_l);
    noc_async_read(read_addr, l1_write_addr, num_tiles_l * page_bytes);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_s);
    noc_async_read(read_addr, l1_write_addr + num_tiles_l * page_bytes, onetile * page_bytes);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_m);
    noc_async_read(read_addr, l1_write_addr + (num_tiles_l + onetile) * page_bytes, onetile * page_bytes);
    noc_async_read_barrier();
    cb_push_back(packet_cb_id, 1);
}
