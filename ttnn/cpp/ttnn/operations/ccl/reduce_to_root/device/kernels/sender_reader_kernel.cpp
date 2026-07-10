// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
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

    Noc noc;

    cb_reserve_back(packet_cb_id, 1);
    uint32_t l1_write_addr = get_write_ptr(packet_cb_id);
    noc.async_read(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(l1_write_addr),
        num_tiles_l * page_bytes,
        {.noc_x = core_noc_x, .noc_y = core_noc_y, .addr = src_addr_l},
        {});
    noc.async_read(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(l1_write_addr + num_tiles_l * page_bytes),
        onetile * page_bytes,
        {.noc_x = core_noc_x, .noc_y = core_noc_y, .addr = src_addr_s},
        {});
    noc.async_read(
        UnicastEndpoint{},
        CoreLocalMem<uint32_t>(l1_write_addr + (num_tiles_l + onetile) * page_bytes),
        onetile * page_bytes,
        {.noc_x = core_noc_x, .noc_y = core_noc_y, .addr = src_addr_m},
        {});
    noc_async_read_barrier();
    cb_push_back(packet_cb_id, 1);
}
