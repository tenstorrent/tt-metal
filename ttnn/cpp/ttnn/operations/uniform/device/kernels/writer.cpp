// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to;
    f2u_from.u = get_arg_val<uint32_t>(1);
    f2u_to.u = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t end_id = start_id + num_tiles;

    float random_range = f2u_to.f - f2u_from.f;
    DPRINT << f2u_from.f << " " << f2u_to.f << ENDL();

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;

    const InterleavedAddrGenFast<true> output_addrg = {
        .bank_base_address = dst_addr,
        .page_size = get_tile_size(cb_id_out0),
        .data_format = get_dataformat(cb_id_out0)};

    uint32_t max_uint = 2147483647;

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t l1_read_ptr = get_read_ptr(cb_id_out0);
        auto l1_addr = reinterpret_cast<uint8_t *>(l1_read_ptr);
        for (uint32_t k = 0; k < 32; k++) {
            for (uint32_t j = 0; j < 32; j++) {
                uint32_t cur = *(uint32_t *)l1_addr;
                *(float *)l1_addr = static_cast<float>(cur) / max_uint * random_range + f2u_from.f;
                l1_addr += 4;
            }
        }
        noc_async_write_tile(i, output_addrg, l1_read_ptr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
