// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "tensix_types.h"

// #include "debug/dprint.h"

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
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t num_readers = get_compile_time_arg_val(2);

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_readers>();
    uint32_t barrier_count = 0;
    uint32_t curr_tile_id = start_id;
    cb_reserve_back(cb_id_in0, block_num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = curr_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc_async_read_tile(tile_id, s, l1_write_addr);
            tile_id++;
            l1_write_addr += tile_bytes;
            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
        l1_write_addr += padded_offset_bytes;
        curr_tile_id += input_width_offset_tiles;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, block_num_tiles);
}
