// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_2;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    Noc noc;
    CircularBuffer cb_out0(cb_id_out0);
    UnicastEndpoint dst;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb_out0.wait_front(ublock_size_tiles);
        noc.async_write(
            cb_out0, dst, ublock_size_bytes, {}, {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = dst_addr});

        noc.async_write_barrier();

        cb_out0.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
