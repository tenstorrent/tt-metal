// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
// #include <unistd.h>

void kernel_main() {
    // Constants for indexing
    constexpr uint8_t x_index = 0;
    constexpr uint8_t y_index = 1;

    size_t rt_arg_idx = 0;

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
    constexpr uint32_t packet_worker_start_x = get_compile_time_arg_val(14);
    constexpr uint32_t packet_worker_start_y = get_compile_time_arg_val(15);
    constexpr uint32_t packet_worker_end_x = get_compile_time_arg_val(16);
    constexpr uint32_t packet_worker_end_y = get_compile_time_arg_val(17);
    constexpr uint32_t num_sender_cores = get_compile_time_arg_val(18);
    constexpr uint32_t total_num_read_txns = get_compile_time_arg_val(19);

    // Derived compile-time constants
    constexpr uint32_t input_tensor_cores = input_shard_cores_per_device * num_devices;
    constexpr uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;

    // Precompute constants for optimization
    constexpr uint32_t bytes_per_tile_group = tiles_per_core_width * page_size_bytes;
    constexpr uint32_t num_dests =
        (packet_worker_end_x - packet_worker_start_x + 1) * (packet_worker_end_y - packet_worker_start_y + 1);
    constexpr uint32_t chip_id_offset = chip_id * num_pages_per_packet * page_size_bytes;

    constexpr uint32_t other_devices = num_devices - 1;
    constexpr uint8_t device_order[other_devices] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
    constexpr uint8_t input_core_xy[input_tensor_cores][2] = INPUT_CORE_XY;
    constexpr uint8_t output_core_xy[output_cores_per_device][2] = OUTPUT_CORE_XY;
    constexpr uint8_t schedule[total_num_read_txns][3] = SCHEDULE;
    constexpr uint32_t total_senders = num_sender_cores * other_devices;

    // Runtime arguments
    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t local_semaphore_address = get_semaphore(get_arg_val<uint32_t>(rt_arg_idx++));
    bool sender_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    bool worker_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t linear_input_packet_start_idx = get_arg_val<uint32_t>(rt_arg_idx++);
    bool receiver_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_shard_start = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_shard_end = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_total_num_pages = get_arg_val<uint32_t>(rt_arg_idx++);

    // Bank base addresses (compute once)
    const uint32_t bank_base_address = get_write_ptr(input_tensor_cb_id);

    uint32_t sender_read_addr = get_write_ptr(fabric_sender_cb_id);

    if (sender_core) {
        for (uint32_t target_device_id : device_order) {
            const uint32_t base_core = target_device_id * input_shard_cores_per_device;

            uint32_t num_pages_read = 0;
            uint32_t num_pages_reserve_push = 0;
            uint32_t shard_idx = sender_shard_start;
            while (num_pages_read < sender_total_num_pages) {
                const uint8_t curr_core = base_core + schedule[shard_idx][0];
                const uint32_t read_offset = schedule[shard_idx][1];
                const uint32_t read_size = schedule[shard_idx][2];
                num_pages_reserve_push += read_size;

                auto num_pages_left = sender_total_num_pages - num_pages_read;
                const uint32_t curr_packet_num_pages = std::min(num_pages_per_packet, num_pages_left);

                const uint32_t x = input_core_xy[curr_core][x_index];
                const uint32_t y = input_core_xy[curr_core][y_index];
                const uint32_t offset_address = bank_base_address + (read_offset * page_size_bytes);
                const uint64_t shard_noc_addr = get_noc_addr(x, y, offset_address);
                const uint32_t transfer_size = read_size * page_size_bytes;

                cb_reserve_back(fabric_sender_cb_id, num_pages_reserve_push);
                noc_async_read(shard_noc_addr, sender_read_addr, transfer_size);

                if (num_pages_reserve_push >= curr_packet_num_pages) {
                    noc_async_read_barrier();
                    cb_push_back(fabric_sender_cb_id, num_pages_reserve_push);
                    num_pages_reserve_push = 0;
                }

                sender_read_addr += transfer_size;
                num_pages_read += read_size;
                shard_idx++;
            }
        }
    } else if (worker_core) {
        // Calculate base addresses once
        const uint32_t base_input_tensor_addr = get_read_ptr(input_tensor_cb_id);
        const uint32_t base_receiver_l1_addresses = get_read_ptr(fabric_receiver_cb_id) + chip_id_offset;

        for (uint32_t i = 0; i < num_pages_per_packet; i++) {
            const uint32_t rem = linear_input_packet_start_idx + i;
            const uint32_t linear_input_core_idcs = rem / tiles_per_core_width;
            const uint32_t linear_input_tile_offsets = rem % tiles_per_core_width;

            if (linear_input_core_idcs >= input_tensor_cores) {
                break;
            }

            const uint32_t core_x = input_core_xy[linear_input_core_idcs][x_index];
            const uint32_t core_y = input_core_xy[linear_input_core_idcs][y_index];
            const uint32_t tile_offset = linear_input_tile_offsets * page_size_bytes;

            const uint64_t output_noc_address = get_noc_addr(core_x, core_y, base_input_tensor_addr + tile_offset);
            const uint32_t receiver_l1_address = base_receiver_l1_addresses + i * page_size_bytes;

            noc_async_read(output_noc_address, receiver_l1_address, page_size_bytes);
        }

        noc_semaphore_wait((uint32_t*)receiver_semaphore_address, other_devices);

        noc_async_read_barrier();
        cb_push_back(fabric_receiver_cb_id, num_pages_per_packet * num_devices);
    }
    noc_semaphore_set((uint32_t*)local_semaphore_address, INVALID);
    noc_semaphore_set((uint32_t*)receiver_semaphore_address, INVALID);
    noc_async_write_barrier();
}
