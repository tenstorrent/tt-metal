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

    constexpr uint32_t input_tensor_cores = input_shard_cores_per_device * num_devices;

    // Calculate the total number of packets needed to send all the tiles
    constexpr uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;

    constexpr uint32_t device_order[num_devices - 1] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
    constexpr uint32_t receiver_core_for_device[num_devices][2] = RECEIVER_CORE_XY;
    constexpr uint32_t input_core_xy[input_tensor_cores][2] = INPUT_CORE_XY;
    constexpr uint32_t output_core_xy[ncores_output][2] = OUTPUT_CORE_XY;

    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t local_semaphore_address = get_semaphore(get_arg_val<uint32_t>(rt_arg_idx++));
    bool sender_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    bool worker_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);

    DPRINT << "receiver_semaphore_address " << receiver_semaphore_address << " local_semaphore_address "
           << local_semaphore_address << ENDL();
    DPRINT << "receiver_semaphore_address value: " << *(uint32_t*)receiver_semaphore_address
           << " local_semaphore_address value: " << *(uint32_t*)local_semaphore_address << ENDL();
    DPRINT << "sender_core: " << (uint32_t)sender_core << " worker_core: " << (uint32_t)worker_core << ENDL();
    DPRINT << "tiles_per_core_width_output: " << tiles_per_core_width_output << ENDL();
    bool receiver_core = true;
    uint32_t receiver_for_device_id = 0;

    uint32_t bank_base_address = get_read_ptr(input_tensor_cb_id);
    uint32_t x_index = 0;
    uint32_t y_index = 1;
    if (sender_core) {
        for (auto target_device_id : device_order) {
            if (target_device_id == chip_id) {
                break;
            }
            uint32_t base_core = target_device_id * input_shard_cores_per_device;
            uint32_t curr_tile = 0;  // this is 0 to tiles_per_core_width - 1

            for (uint32_t curr_core = base_core; curr_core < base_core + input_shard_cores_per_device; ++curr_core) {
                uint32_t x = input_core_xy[curr_core][x_index];
                uint32_t y = input_core_xy[curr_core][y_index];
                uint64_t shard_noc_addr = get_noc_addr(x, y, bank_base_address);
                DPRINT << "Reserving back " << tiles_per_core_width << " tiles for fabric sender cb" << ENDL();
                cb_reserve_back(fabric_sender_cb_id, tiles_per_core_width);
                DPRINT << "Reserving back done" << ENDL();
                uint32_t sender_read_addr = get_read_ptr(fabric_sender_cb_id);
                noc_async_read(shard_noc_addr, sender_read_addr, tiles_per_core_width * page_size_bytes);
                noc_async_read_barrier();
                print_tiles(fabric_sender_cb_id, 0, tiles_per_core_width, true);
                DPRINT << "Pushing back " << tiles_per_core_width << " tiles to fabric sender cb" << ENDL();
                cb_push_back(fabric_sender_cb_id, tiles_per_core_width);
            }
            DPRINT << "Finished pushing back all tiles to fabric sender cb for device " << target_device_id << ENDL();
        }
        DPRINT << "Finished pushing back all tiles to fabric sender cb" << ENDL();
    } else if (worker_core) {
        DPRINT << "accumulator addr " << get_noc_addr(get_read_ptr(accumulator_cb_id))
               << " local semaphore noc addr: " << get_noc_addr(local_semaphore_address) << ENDL();
        noc_semaphore_wait((uint32_t*)local_semaphore_address, num_devices - 1);
        // while (*(uint32_t*)local_semaphore_address != num_devices - 1) {
        //     // Wait for the semaphore to be set
        //     DPRINT << "Waiting for local semaphore to be set, current value: " <<
        //     *(uint32_t*)(local_semaphore_address)
        //            << ENDL();
        // }
        for (uint32_t target_device_id = 0; target_device_id < num_devices; ++target_device_id) {
            if (target_device_id == chip_id) {
                continue;
            }
            print_full_tile(accumulator_cb_id, target_device_id * tiles_per_core_width_output, true);
        }
        DPRINT << "Pushing back " << tiles_per_core_width_output * num_devices << " tiles to accumulator cb" << ENDL();
        cb_push_back(accumulator_cb_id, tiles_per_core_width_output * num_devices);
        *(uint32_t*)local_semaphore_address = 0;
    } else {
        // Do nothing
        // win
    }
}
