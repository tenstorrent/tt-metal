// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src0_noc_y = get_arg_val<uint32_t>(2);
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(3);
    uint32_t src1_addr = get_arg_val<uint32_t>(4);
    uint32_t src1_noc_x = get_arg_val<uint32_t>(5);
    uint32_t src1_noc_y = get_arg_val<uint32_t>(6);
    uint32_t src1_num_tiles = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);
    uint32_t ublock_size_tiles = 1;

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in1(cb_id_in1);
    UnicastEndpoint src;

    uint32_t num_tiles = src0_num_tiles > src1_num_tiles ? src0_num_tiles : src1_num_tiles;

    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        if (i < src0_num_tiles) {
            cb_in0.reserve_back(ublock_size_tiles);

            noc.async_read(
                src, cb_in0, ublock_size_bytes_0, {.noc_x = src0_noc_x, .noc_y = src0_noc_y, .addr = src0_addr}, {});

            noc.async_read_barrier();

            cb_in0.push_back(ublock_size_tiles);

            src0_addr += ublock_size_bytes_0;
        }

        if (i < src1_num_tiles) {
            cb_in1.reserve_back(ublock_size_tiles);

            noc.async_read(
                src, cb_in1, ublock_size_bytes_1, {.noc_x = src1_noc_x, .noc_y = src1_noc_y, .addr = src1_addr}, {});

            noc.async_read_barrier();

            cb_in1.push_back(ublock_size_tiles);

            src1_addr += ublock_size_bytes_1;
        }
    }
}
