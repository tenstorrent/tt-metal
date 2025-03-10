// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

inline void print_tiles(uint32_t cb_id, uint32_t tile_start = 0, uint32_t num_tiles = 1, bool untilize = false) {
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        print_full_tile(cb_id, tile_start + tile_idx, untilize);
    }
}

void kernel_main() {
    DPRINT << "Starting kernel_main for reader" << ENDL();
    size_t ct_arg_idx = 0, rt_arg_idx = 0;

    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t fabric_sender_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_core_width = get_compile_time_arg_val(3);
    constexpr uint32_t cores_per_device = get_compile_time_arg_val(4);
    constexpr uint32_t num_devices = get_compile_time_arg_val(5);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(6);

    constexpr uint32_t device_order[num_devices - 1] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines

    typedef ShardedInfo<
        get_compile_time_arg_val(7),   // Memory layout
        get_compile_time_arg_val(8),   // The number of sharding cores
        get_compile_time_arg_val(9),   // The page size we offset each write to
        get_compile_time_arg_val(10),  // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(11),  // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(12),  // pages_per_shard_x
        get_compile_time_arg_val(13)>  // pages_per_shard_y
        tensor_shard_info;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt_arg_idx));
    rt_arg_idx += rt_increment;
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {
        .bank_base_address = get_read_ptr(input_tensor_cb_id), .shard_array = mapping_table};

    for (auto target_device_id : device_order) {
        if (target_device_id == chip_id) {
            break;
        }
        uint32_t base_page = target_device_id * cores_per_device * tiles_per_core_width;
        uint32_t page_offset = base_page;
        uint32_t target_device_core_offset = target_device_id * cores_per_device;
        for (uint32_t core_id = 0; core_id < cores_per_device; ++core_id) {
            const auto [noc_addr, num_pages] = s0.get_contiguous_noc_addr(page_offset);  // might as well use this as
            cb_reserve_back(fabric_sender_cb_id, num_pages);
            noc_async_read(noc_addr, get_read_ptr(fabric_sender_cb_id), num_pages * page_size_bytes);

            page_offset += num_pages;
            noc_async_read_barrier();
            print_tiles(fabric_sender_cb_id, 0, num_pages, true);
            DPRINT << "Pushing back " << num_pages << " tiles to fabric sender cb" << ENDL();
            cb_push_back(fabric_sender_cb_id, num_pages);
        }
    }
}
