// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    DPRINT << "sender reader kernel started\n";
    const uint32_t src_addr_l = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles_l = get_arg_val<uint32_t>(1);
    const uint32_t src_addr_s = get_arg_val<uint32_t>(2);
    const uint32_t src_addr_m = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_id_in_l = 0;
    constexpr uint32_t cb_id_in_s = 1;
    constexpr uint32_t cb_id_in_m = 2;

    // ublocks size defined in tiles
    // l is [8, 256] so read all 8 tiles to the cb at once : might be different so keeping it general now
    // m and s are [1, 8] so read one tile to the cb
    constexpr uint32_t onetile = 1;

    const uint32_t page_bytes = get_arg_val<uint32_t>(4);  // should be the size of a tiny tile page
    const uint32_t core_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(6);

    DPRINT << "num tiles l: " << (uint32_t)num_tiles_l << "\n";
    // for tensor l
    uint64_t read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_l);
    DPRINT << "src adr l: " << (uint32_t)src_addr_l << "\n";
    DPRINT << "read addr l: " << (uint64_t)read_addr << "\n";
    DPRINT << "reserving cb id: " << (uint32_t)cb_id_in_l << "\n";
    for (uint32_t i = 0; i < num_tiles_l; ++i) {
        cb_reserve_back(cb_id_in_l, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in_l);
        noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
        read_addr += onetile * page_bytes;
        noc_async_read_barrier();
        cb_push_back(cb_id_in_l, onetile);
    }

    // for tensor s
    cb_reserve_back(cb_id_in_s, onetile);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in_s);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_s);
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in_s, onetile);

    // for tensor m
    cb_reserve_back(cb_id_in_m, onetile);
    l1_write_addr = get_write_ptr(cb_id_in_m);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_m);
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in_m, onetile);
    DPRINT << "sender reader kernel completed\n";
}
