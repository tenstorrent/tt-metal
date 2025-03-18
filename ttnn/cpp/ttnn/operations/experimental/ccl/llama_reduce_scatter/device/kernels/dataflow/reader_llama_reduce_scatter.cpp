// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

// inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     DPRINT << "===" << tile_id << "===" << ENDL();
//     for (uint16_t r = 0; r < 32; ++r) {
//         DPRINT << (uint)r << " : "
//                << TileSlice(
//                       cb_id,
//                       tile_id,
//                       SliceRange{
//                           .h0 = (uint8_t)r,
//                           .h1 = (uint8_t)(r + 1),
//                           .hs = (uint8_t)1,
//                           .w0 = (uint8_t)0,
//                           .w1 = (uint8_t)32,
//                           .ws = (uint8_t)1},
//                       true,
//                       untilize)
//                << ENDL();
//     }
//     DPRINT << "++++++" << ENDL();
// }

// inline void print_tiles(uint32_t cb_id, uint32_t tile_start = 0, uint32_t num_tiles = 1, bool untilize = false) {
//     for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
//         print_full_tile(cb_id, tile_start + tile_idx, untilize);
//     }
// }

void kernel_main() {
    // DPRINT << "Starting kernel_main for reader" << ENDL();
    size_t ct_arg_idx = 0, rt_arg_idx = 0;

    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t fabric_sender_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t fabric_receiver_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t output_tensor_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t chip_id = get_compile_time_arg_val(6);
    constexpr uint32_t tiles_per_core_width = get_compile_time_arg_val(7);
    constexpr uint32_t tiles_per_core_width_output = get_compile_time_arg_val(8);
    constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(9);
    constexpr uint32_t input_shard_cores_per_device = get_compile_time_arg_val(10);
    constexpr uint32_t num_devices = get_compile_time_arg_val(11);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t ncores_output = get_compile_time_arg_val(13);

    // Derived compile-time constants
    constexpr uint32_t input_tensor_cores = input_shard_cores_per_device * num_devices;
    constexpr uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;

    constexpr uint8_t device_order[num_devices - 1] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
    constexpr uint8_t receiver_core_for_device[num_devices][2] = RECEIVER_CORE_XY;
    constexpr uint8_t input_core_xy[input_tensor_cores][2] = INPUT_CORE_XY;
    constexpr uint8_t output_core_xy[ncores_output][2] = OUTPUT_CORE_XY;

    // Runtime arguments
    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t local_semaphore_address = get_semaphore(get_arg_val<uint32_t>(rt_arg_idx++));
    bool sender_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    bool worker_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t linear_input_page_idx = get_arg_val<uint32_t>(rt_arg_idx++);

    // Constants for indexing
    constexpr uint32_t x_index = 0;
    constexpr uint32_t y_index = 1;

    // Constants for memory operations
    uint32_t bank_base_address = get_write_ptr(input_tensor_cb_id);

    if (sender_core) {
        // DPRINT << "input_shard_cores_per_device " << input_shard_cores_per_device << ENDL();
        for (auto target_device_id : device_order) {
            if (target_device_id == chip_id) {
                break;
            }

            uint32_t base_core = target_device_id * input_shard_cores_per_device;
            // DPRINT << "base_core " << base_core << " input_shard_cores_per_device " << input_shard_cores_per_device
            //        << ENDL();

            // Precompute the number of bytes to transfer per tile group
            constexpr uint32_t bytes_per_tile_group = tiles_per_core_width * page_size_bytes;

            for (uint32_t curr_core = base_core; curr_core < base_core + input_shard_cores_per_device; ++curr_core) {
                uint32_t x = input_core_xy[curr_core][x_index];
                uint32_t y = input_core_xy[curr_core][y_index];
                uint64_t shard_noc_addr = get_noc_addr(x, y, bank_base_address);
                // DPRINT << "reserving " << tiles_per_core_width << " pages" << ENDL();
                cb_reserve_back(fabric_sender_cb_id, tiles_per_core_width);
                uint32_t sender_read_addr = get_write_ptr(fabric_sender_cb_id);
                noc_async_read(shard_noc_addr, sender_read_addr, bytes_per_tile_group);
                noc_async_read_barrier();
                // print_tiles(fabric_sender_cb_id, 0, tiles_per_core_width, true);
                cb_push_back(fabric_sender_cb_id, tiles_per_core_width);
            }
        }
    } else if (worker_core) {
        // DPRINT << "linear_input_page_idx " << linear_input_page_idx << ENDL();
        // Calculate core index and tile offset once
        uint32_t linear_input_core_idx = linear_input_page_idx / tiles_per_core_width;
        // DPRINT << "linear_input_core_idx " << linear_input_core_idx << ENDL();
        uint32_t linear_input_tile_offset = linear_input_page_idx % tiles_per_core_width;
        // DPRINT << "linear_input_tile_offset " << linear_input_tile_offset << ENDL();

        uint32_t core_x = input_core_xy[linear_input_core_idx][x_index];
        uint32_t core_y = input_core_xy[linear_input_core_idx][y_index];

        uint32_t tile_offset = linear_input_tile_offset * page_size_bytes;
        uint64_t tile_addr = get_noc_addr(core_x, core_y, get_read_ptr(input_tensor_cb_id) + tile_offset);

        // Precompute the destination address
        uint32_t dest_addr = get_write_ptr(accumulator_cb_id) + chip_id * page_size_bytes;

        noc_async_read(tile_addr, dest_addr, page_size_bytes);
        noc_async_read_barrier();
        // DPRINT << "waiting for semaphore" << ENDL();
        noc_semaphore_wait((uint32_t*)local_semaphore_address, num_devices - 1);
        // DPRINT << "semaphore received" << ENDL();
        // Push all accumulated data at once
        cb_push_back(accumulator_cb_id, tiles_per_core_width_output * num_devices);
        *(uint32_t*)local_semaphore_address = 0;
    } else {
        // Do nothing
        // win
    }
    // DPRINT << "Kernel finished" << ENDL();
}
