// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_height = 32;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles_per_block = get_arg_val<uint32_t>(3);
    const uint32_t block_width_size = get_arg_val<uint32_t>(4);
    const uint32_t num_full_blocks_in_row = get_arg_val<uint32_t>(5);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(8);

    constexpr auto src_tensor_args = TensorAccessorArgs<1>();

    const auto s = TensorAccessor(src_tensor_args, src_addr);

    experimental::CircularBuffer cb(cb_id_in0);
    experimental::Noc noc;

    uint32_t base_stick_ids[tile_height];
    uint32_t base_offsets[tile_height];

    auto read_tiles = [&](const uint32_t& num_tiles, const uint32_t& width_size) {
        cb.reserve_back(num_tiles);
        uint32_t l1_write_offset = 0;
        for (uint32_t k = 0; k < tile_height; k++) {
            noc.async_read(
                s,
                cb,
                width_size,
                {.page_id = base_stick_ids[k], .offset_bytes = base_offsets[k]},
                {.offset_bytes = l1_write_offset});
            l1_write_offset += width_size;
            base_offsets[k] += width_size;
        }
        noc.async_read_barrier();
        cb.push_back(num_tiles);
    };

    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
        // Get Base Addresses
        for (uint32_t j = 0; j < tile_height; j++) {
            base_src_noc_addr[j] = s.get_noc_addr(stick_id);
            stick_id++;
        }

        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, block_width_size);
        }
    }
}
