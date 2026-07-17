// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_height = get_compile_time_arg_val(1);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_per_block = get_arg_val<uint32_t>(3);
    const uint32_t block_width_size = get_arg_val<uint32_t>(4);
    const uint32_t num_full_blocks_in_row = get_arg_val<uint32_t>(5);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(8);

    constexpr auto src_tensor_args = TensorAccessorArgs<2>();

    const auto s = TensorAccessor(src_tensor_args, src_addr);

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);

    uint32_t stick_ids[tile_height];
    uint32_t stick_offset = 0;

    auto read_tiles = [&](const uint32_t& num_tiles, const uint32_t& width_size) {
        cb_in0.reserve_back(num_tiles);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                s, dst, width_size, {.page_id = stick_ids[k], .offset_bytes = stick_offset}, {.offset_bytes = 0});
            l1_write_addr += width_size;
        }
        stick_offset += width_size;
        noc.async_read_barrier();
        cb_in0.push_back(num_tiles);
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
