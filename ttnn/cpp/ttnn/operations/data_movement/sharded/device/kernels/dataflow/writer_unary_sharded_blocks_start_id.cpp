// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t padded_offset = get_arg_val<uint32_t>(3);
    const uint32_t block_width_padded_num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t output_width_tiles = get_arg_val<uint32_t>(5);
    const uint32_t start_id_offset = get_arg_val<uint32_t>(6);
    const uint32_t start_id_base = get_arg_val<uint32_t>(7);

    // compile-time args
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(1),   // Memory layout
        get_compile_time_arg_val(2),   // The number of sharding cores
        get_compile_time_arg_val(3),   // The page size we offset each write to
        get_compile_time_arg_val(4),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(5),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(6),   // pages_per_shard_x
        get_compile_time_arg_val(7)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(8));
    experimental::ShardedAddrGen<tensor_shard_info> s = {.bank_base_address = dst_addr, .shard_array = mapping_table};

    uint32_t row_start_tile_id = start_id_base + start_id_offset;
    cb_wait_front(cb_id_out, block_width_padded_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            uint64_t dst_noc_addr = get_noc_addr(tile_id, s);
            noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
            tile_id++;
            l1_read_addr += tile_bytes;
        }
        l1_read_addr += padded_offset;
        row_start_tile_id += output_width_tiles;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, block_width_padded_num_tiles);
}
