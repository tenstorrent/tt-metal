// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pass2_writer.cpp — BRISC1 / writer for device-side Stockham Pass 2.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "pass2_common.h"

void kernel_main() {
    const uint32_t out_r_addr  = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr  = get_arg_val<uint32_t>(1);
    const uint32_t first_tile  = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(3);

    const DataFormat df = get_dataformat(CB_B_R);
    const uint32_t   ts = get_tile_size(CB_B_R);

    InterleavedAddrGenFast<true> out_r_gen = {
        .bank_base_address = out_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> out_i_gen = {
        .bank_base_address = out_i_addr, .page_size = ts, .data_format = df};

    for (uint32_t k = 0; k < num_tiles; ++k) {
        const uint32_t tile_idx = first_tile + k;

        cb_wait_front(CB_B_R, 1);
        cb_wait_front(CB_B_I, 1);

        noc_async_write_tile(tile_idx, out_r_gen, get_read_ptr(CB_B_R));
        noc_async_write_tile(tile_idx, out_i_gen, get_read_ptr(CB_B_I));
        noc_async_write_barrier();

        cb_pop_front(CB_B_R, 1);
        cb_pop_front(CB_B_I, 1);
    }
}
