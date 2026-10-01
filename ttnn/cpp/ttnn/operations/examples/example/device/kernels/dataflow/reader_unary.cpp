// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_noc_y = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
    UnicastEndpoint src;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb_in0.reserve_back(ublock_size_tiles);
        noc.async_read(src, cb_in0, ublock_size_bytes, {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = src_addr}, {});

        noc.async_read_barrier();

        cb_in0.push_back(ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}
