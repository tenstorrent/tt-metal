// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
constexpr bool flush = false;

template <bool ring_topology>
FORCE_INLINE uint32_t distance(uint32_t chip_id, uint32_t target_device_id, uint32_t num_devices) {
    if constexpr (ring_topology) {
        uint32_t line_distance = std::abs(int(target_device_id) - int(chip_id));
        return std::min(line_distance, num_devices - line_distance);
    } else {
        return std::abs(int(target_device_id) - int(chip_id));
    }
}

template <bool ring_topology>
FORCE_INLINE tt::tt_fabric::WorkerToFabricEdmSender& get_fabric_connection(
    FabricConnectionManager& fabric_connection, uint32_t chip_id, uint32_t target_device_id, uint32_t num_devices) {
    if constexpr (ring_topology) {
        uint32_t right_distance = (target_device_id - chip_id + num_devices) % num_devices;
        uint32_t left_distance = (chip_id - target_device_id + num_devices) % num_devices;
        if (right_distance <= left_distance) {
            return fabric_connection.get_forward_connection();
        } else {
            return fabric_connection.get_backward_connection();
        }
    } else {
        return target_device_id > chip_id ? fabric_connection.get_forward_connection()
                                          : fabric_connection.get_backward_connection();
    }
}

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
    constexpr uint32_t packet_receiver_core_x = get_compile_time_arg_val(14);
    constexpr uint32_t packet_receiver_core_y = get_compile_time_arg_val(15);
    constexpr uint32_t num_packet_worker_cores = get_compile_time_arg_val(16);
    constexpr bool ring_topology = (bool)get_compile_time_arg_val(17);
    // Derived compile-time constants
    constexpr uint32_t input_tensor_cores = input_shard_cores_per_device * num_devices;
    constexpr uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;
    constexpr uint32_t last_packet_num_pages =
        (input_shard_cores_per_device * tiles_per_core_width % num_pages_per_packet == 0)
            ? num_pages_per_packet
            : input_shard_cores_per_device * tiles_per_core_width % num_pages_per_packet;
    constexpr size_t packet_header_size = sizeof(PACKET_HEADER_TYPE);

    // Precomputed constants for better optimization
    constexpr uint32_t chip_id_offset = chip_id * num_pages_per_packet * page_size_bytes;

    constexpr uint32_t other_devices = num_devices - 1;

    // Runtime arguments
    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t local_semaphore_address = get_semaphore(get_arg_val<uint32_t>(rt_arg_idx++));
    bool sender_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    bool worker_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t linear_output_page_start_idx = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_packet_start = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_packet_end = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t sender_total_num_pages = get_arg_val<uint32_t>(rt_arg_idx++);
    if (sender_core) {
        auto fabric_connection =
            FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
                rt_arg_idx);
        // Set up packet headers once
        constexpr uint8_t device_order[other_devices] =
            DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
        constexpr uint8_t packet_worker_cores[num_packet_worker_cores][2] = PACKET_WORKER_CORES;
        const auto packet_header_buffer_addr = get_read_ptr(packet_header_cb_id);
        auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
        auto* sem_inc_packet_header =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr + packet_header_size);
        const uint64_t sem_noc_addr =
            safe_get_noc_addr(packet_receiver_core_x, packet_receiver_core_y, receiver_semaphore_address, 0);
        sem_inc_packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            sem_noc_addr,
            static_cast<uint16_t>(1),  // increment 1
            32});

        const uint32_t base_receiver_l1_addr = get_read_ptr(fabric_receiver_cb_id);

        // Precompute the packet offset once
        const uint32_t packet_offset = base_receiver_l1_addr + chip_id_offset;

        if (fabric_connection.is_logically_connected()) {
            fabric_connection.open_finish();
        }
        for (uint32_t target_device_id : device_order) {
            // Calculate device-specific constants once per device
            const uint32_t num_hops = distance<ring_topology>(chip_id, target_device_id, num_devices);
            unicast_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
            auto& fabric_conn =
                get_fabric_connection<ring_topology>(fabric_connection, chip_id, target_device_id, num_devices);

            uint32_t num_pages_sent = 0;
            uint32_t packet = sender_packet_start;
            while (num_pages_sent < sender_total_num_pages) {
                // Determine packet size based on whether it's the last packet
                auto num_pages_left = sender_total_num_pages - num_pages_sent;
                const uint32_t curr_packet_num_pages = std::min(num_pages_per_packet, num_pages_left);
                const uint32_t curr_packet_size_bytes = curr_packet_num_pages * page_size_bytes;

                const uint32_t receiver_core_x = packet_worker_cores[packet][x_index];
                const uint32_t receiver_core_y = packet_worker_cores[packet][y_index];
                const uint64_t noc0_dest_noc_addr =
                    safe_get_noc_addr(receiver_core_x, receiver_core_y, packet_offset, 0);

                cb_wait_front(fabric_sender_cb_id, curr_packet_num_pages);
                const auto sender_l1_addr = get_read_ptr(fabric_sender_cb_id);

                const uint64_t sem_noc_addr =
                    safe_get_noc_addr(receiver_core_x, receiver_core_y, receiver_semaphore_address, 0);
                unicast_packet_header->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                        noc0_dest_noc_addr, sem_noc_addr, 1, 32, flush),
                    curr_packet_size_bytes);

                fabric_conn.wait_for_empty_write_slot();

                fabric_conn.send_payload_without_header_non_blocking_from_address(
                    sender_l1_addr, curr_packet_size_bytes);

                fabric_conn.send_payload_flush_blocking_from_address(
                    (uint32_t)unicast_packet_header, packet_header_size);

                cb_pop_front(fabric_sender_cb_id, curr_packet_num_pages);

                num_pages_sent += curr_packet_num_pages;
                packet++;
            }
        }

        if (fabric_connection.is_logically_connected()) {
            fabric_connection.close();
        }
    } else if (worker_core) {
#ifndef SKIP_WRITE_BACK
        constexpr uint8_t output_core_xy[output_cores_per_device][2] = OUTPUT_CORE_XY;
        uint64_t noc_addresses[num_pages_per_packet];
        uint32_t accumulator_l1_addresses[num_pages_per_packet];
        uint32_t output_tensor_base_addr = get_read_ptr(output_tensor_cb_id);
        auto accumulator_l1_addr = get_read_ptr(accumulator_cb_id);

        uint32_t num_packets = num_pages_per_packet;
        for (uint32_t i = 0; i < num_pages_per_packet; i++) {
            uint32_t rem = linear_output_page_start_idx + i;
            uint32_t linear_output_core_idcs = rem / tiles_per_core_width_output;
            if (linear_output_core_idcs >= output_cores_per_device) {
                num_packets = i;
                break;
            }
            uint32_t linear_output_tile_offsets = rem % tiles_per_core_width_output;
            noc_addresses[i] = get_noc_addr(
                output_core_xy[linear_output_core_idcs][x_index],
                output_core_xy[linear_output_core_idcs][y_index],
                output_tensor_base_addr + (linear_output_tile_offsets * page_size_bytes));
            accumulator_l1_addresses[i] = accumulator_l1_addr + i * page_size_bytes;
        }

        cb_wait_front(accumulator_cb_id, num_pages_per_packet);

        // Process all tiles
        for (uint32_t tile = 0; tile < num_packets; tile++) {
            noc_async_write(accumulator_l1_addresses[tile], noc_addresses[tile], page_size_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(accumulator_cb_id, num_pages_per_packet);
#else
        cb_wait_front(output_tensor_cb_id, num_pages_per_packet);
#endif
    }
}
