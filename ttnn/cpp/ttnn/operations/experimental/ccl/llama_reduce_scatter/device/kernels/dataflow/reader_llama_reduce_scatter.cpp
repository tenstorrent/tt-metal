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

template <uint8_t noc_ind = noc_index>
FORCE_INLINE std::uint64_t static_noc_multicast_addr(
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t addr) {
    if constexpr (noc_ind == 0) {
        return get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr);
    } else {
        return get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, addr);
    }
}

void kernel_main() {
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
    constexpr uint32_t output_cores_per_device = get_compile_time_arg_val(13);
    constexpr uint32_t noc_start_x = get_compile_time_arg_val(14);
    constexpr uint32_t noc_start_y = get_compile_time_arg_val(15);
    constexpr uint32_t noc_end_x = get_compile_time_arg_val(16);
    constexpr uint32_t noc_end_y = get_compile_time_arg_val(17);

    // Derived compile-time constants
    constexpr uint32_t input_tensor_cores = input_shard_cores_per_device * num_devices;
    constexpr uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;

    constexpr uint8_t device_order[num_devices - 1] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
    constexpr uint8_t input_core_xy[input_tensor_cores][2] = INPUT_CORE_XY;
    constexpr uint8_t output_core_xy[output_cores_per_device][2] = OUTPUT_CORE_XY;
    // constexpr uint8_t packet_worker_cores[num_packets_total_per_device][2] = PACKET_WORKER_CORES;

    constexpr uint32_t num_dests = (noc_end_x - noc_start_x + 1) * (noc_end_y - noc_start_y + 1);
    // Runtime arguments
    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t local_semaphore_address = get_semaphore(get_arg_val<uint32_t>(rt_arg_idx++));
    bool sender_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    bool worker_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t linear_input_packet_start_idx = get_arg_val<uint32_t>(rt_arg_idx++);
    bool receiver_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t start_device_idx = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t end_device_idx = get_arg_val<uint32_t>(rt_arg_idx++);

    // Constants for indexing
    constexpr uint8_t x_index = 0;
    constexpr uint8_t y_index = 1;

    // Constants for memory operations
    uint32_t bank_base_address = get_write_ptr(input_tensor_cb_id);

    if (sender_core) {
        // Precompute the number of bytes to transfer per tile group
        constexpr uint32_t bytes_per_tile_group = tiles_per_core_width * page_size_bytes;
        for (uint32_t device_idx = start_device_idx; device_idx < end_device_idx; device_idx++) {
            uint32_t target_device_id = device_order[device_idx];
            if (target_device_id == chip_id) {
                continue;
            }

            uint32_t base_core = target_device_id * input_shard_cores_per_device;

            for (uint32_t curr_core = base_core; curr_core < base_core + input_shard_cores_per_device; ++curr_core) {
                uint32_t x = input_core_xy[curr_core][x_index];
                uint32_t y = input_core_xy[curr_core][y_index];
                uint64_t shard_noc_addr = get_noc_addr(x, y, bank_base_address);
                //
                cb_reserve_back(fabric_sender_cb_id, tiles_per_core_width);
                uint32_t sender_read_addr = get_write_ptr(fabric_sender_cb_id);
                noc_async_read(shard_noc_addr, sender_read_addr, bytes_per_tile_group);
                noc_async_read_barrier();
                cb_push_back(fabric_sender_cb_id, tiles_per_core_width);
            }
        }
    } else if (worker_core) {
        // Calculate core index and tile offset once
        uint32_t linear_input_core_idcs = 0;
        uint32_t linear_input_tile_offsets = 0;
        uint32_t base_input_tensor_addr = get_read_ptr(input_tensor_cb_id);
        uint64_t output_noc_addresses[num_pages_per_packet];
        uint32_t receiver_l1_addresses[num_pages_per_packet];
        uint32_t base_receiver_l1_addresses =
            get_read_ptr(fabric_receiver_cb_id) + chip_id * num_pages_per_packet * page_size_bytes;

        for (uint32_t i = 0; i < num_pages_per_packet; i++) {
            uint32_t rem = linear_input_packet_start_idx + i;
            linear_input_core_idcs = rem / tiles_per_core_width;
            linear_input_tile_offsets = rem % tiles_per_core_width;
            output_noc_addresses[i] = get_noc_addr(
                input_core_xy[linear_input_core_idcs][x_index],
                input_core_xy[linear_input_core_idcs][y_index],
                base_input_tensor_addr + (linear_input_tile_offsets * page_size_bytes));
            receiver_l1_addresses[i] = base_receiver_l1_addresses + i * page_size_bytes;
        }

        for (uint32_t i = 0; i < num_pages_per_packet; i++) {
            noc_async_read(output_noc_addresses[i], receiver_l1_addresses[i], page_size_bytes);
        }
        if (receiver_core) {
            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t multicast_semaphore_addr =
                static_noc_multicast_addr(noc_start_x, noc_start_y, noc_end_x, noc_end_y, local_semaphore_address);

            noc_semaphore_wait((uint32_t*)receiver_semaphore_address, num_devices - 1);
            //
            noc_semaphore_set_multicast(
                receiver_semaphore_address,
                multicast_semaphore_addr,
                num_dests);  // could do different mcast for each device by having num_devices - 1 receiver cores

            noc_async_atomic_barrier();
        } else {
            noc_semaphore_wait((uint32_t*)local_semaphore_address, (num_devices - 1));
        }
        noc_async_read_barrier();
        cb_push_back(fabric_receiver_cb_id, num_pages_per_packet * num_devices);
        *(uint32_t*)local_semaphore_address = 0;
        *(uint32_t*)receiver_semaphore_address = 0;
    }
}
