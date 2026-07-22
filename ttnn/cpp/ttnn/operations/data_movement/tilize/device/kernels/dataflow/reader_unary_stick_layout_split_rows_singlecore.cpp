// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "experimental/kernel_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t tile_height = 32;

    const auto num_sticks = get_arg(args::num_sticks);
    const auto num_tiles_per_block = get_arg(args::num_tiles_per_block);
    const auto block_width_size = get_arg(args::block_width_size);
    const auto num_full_blocks_in_row = get_arg(args::num_full_blocks_in_row);
    const auto start_stick_id = get_arg(args::start_stick_id);

    const auto s = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);

    uint32_t stick_ids[tile_height];
    uint32_t stick_offset = 0;

    auto read_tiles = [&](const uint32_t& num_tiles, const uint32_t& width_size) {
        dfb_in0.reserve_back(num_tiles);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                s, dst, width_size, {.page_id = stick_ids[k], .offset_bytes = stick_offset}, {.offset_bytes = 0});
            l1_write_addr += width_size;
        }
        stick_offset += width_size;
        noc.async_read_barrier();
        dfb_in0.push_back(num_tiles);
    };

    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
        // Get Base IDs
        for (uint32_t j = 0; j < tile_height; j++) {
            stick_ids[j] = stick_id;
            stick_id++;
        }
        stick_offset = 0;

        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, block_width_size);
        }
    }
}
