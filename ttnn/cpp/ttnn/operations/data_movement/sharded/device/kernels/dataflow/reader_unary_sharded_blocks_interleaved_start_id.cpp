// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "tensix_types.h"

// #include "api/debug/dprint.h"

// Target 8KB of data before a single barrier for 8x8 grid of readers
template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t padded_offset_bytes = get_arg_val<uint32_t>(3);       // input width in tiles - block width in tiles
    const uint32_t input_width_offset_tiles = get_arg_val<uint32_t>(4);  // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(5);           // block_height_tiles * block_width_tiles
    const uint32_t start_id_offset = get_arg_val<uint32_t>(6);
    const uint32_t start_id_base = get_arg_val<uint32_t>(7);
    const uint32_t start_id = start_id_base + start_id_offset;

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t num_readers = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in(cb_id_in0);
    const auto s = TensorAccessor(src_args, src_addr);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_readers>();
    uint32_t barrier_count = 0;
    uint32_t curr_tile_id = start_id;
    uint32_t l1_offset = 0;
    cb_in.reserve_back(block_num_tiles);
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = curr_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc.async_read(s, cb_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = l1_offset});
            tile_id++;
            l1_offset += tile_bytes;
            if (++barrier_count == barrier_threshold) {
                noc.async_read_barrier();
                barrier_count = 0;
            }
        }
        l1_offset += padded_offset_bytes;
        curr_tile_id += input_width_offset_tiles;
    }
    noc.async_read_barrier();
    cb_in.push_back(block_num_tiles);
}
