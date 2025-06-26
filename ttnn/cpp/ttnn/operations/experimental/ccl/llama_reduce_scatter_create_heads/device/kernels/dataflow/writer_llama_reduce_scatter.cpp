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
void kernel_main() {
    // Constants for indexing
    constexpr uint8_t x_index = 0;
    constexpr uint8_t y_index = 1;
    constexpr uint8_t q_heads = 8;
    constexpr uint32_t num_sticks_per_block = 8;
    constexpr uint32_t stick_size_byte = 64 * 2;
    constexpr uint32_t head_dim_bytes = 128 * 2;
    constexpr uint32_t num_packet_headers_storable = 8;
    constexpr uint32_t buffering_factor = 2;
    size_t rt_arg_idx = 0;

    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t fabric_sender_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t fabric_receiver_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(4);
    // constexpr uint32_t output_tensor_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t chip_id = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_core_width = get_compile_time_arg_val(6);
    constexpr uint32_t tiles_per_core_width_output = get_compile_time_arg_val(7);
    constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(8);
    constexpr uint32_t input_shard_cores_per_device = get_compile_time_arg_val(9);
    constexpr uint32_t num_devices = get_compile_time_arg_val(10);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t output_cores_per_device = get_compile_time_arg_val(12);
    constexpr uint32_t packet_receiver_core_x = get_compile_time_arg_val(13);
    constexpr uint32_t packet_receiver_core_y = get_compile_time_arg_val(14);
    constexpr uint32_t num_packet_worker_cores = get_compile_time_arg_val(15);
    constexpr bool RING_TOPOLOGY = get_compile_time_arg_val(16) == 0 ? false : true;

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
    uint32_t q_base_addr = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t k_base_addr = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t v_base_addr = get_arg_val<uint32_t>(rt_arg_idx++);

    if (sender_core) {
        auto fabric_connection =
            FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
                rt_arg_idx);
        // Set up packet headers once
        constexpr uint8_t device_order[other_devices] =
            DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
        constexpr uint8_t packet_worker_cores[num_packet_worker_cores][2] = PACKET_WORKER_CORES;
        cb_reserve_back(packet_header_cb_id, buffering_factor * num_packet_headers_storable);
        const auto packet_header_buffer_addr = get_read_ptr(packet_header_cb_id);
        auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
        auto* sem_inc_packet_header =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr + packet_header_size);
        const uint64_t sem_noc_addr =
            get_noc_addr(packet_receiver_core_x, packet_receiver_core_y, receiver_semaphore_address, 0);
        sem_inc_packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            sem_noc_addr,
            static_cast<uint16_t>(1),  // increment 1
            32});

        const uint32_t base_receiver_l1_addr = get_read_ptr(fabric_receiver_cb_id);

        // Precompute the packet offset once
        const uint32_t packet_offset = base_receiver_l1_addr + chip_id_offset;

        ASSERT(fabric_connection.is_logically_connected());
        fabric_connection.open_finish();
        for (uint32_t target_device_id : device_order) {
            // Calculate device-specific constants once per device
            bool forward_connection = true;
            if constexpr (RING_TOPOLOGY) {
                const int diff = int(target_device_id) - int(chip_id);
                const uint32_t num_hops = std::abs(diff) == 2 ? 2 : 1;
                // To evenly distribute the load, we use the following logic:
                if (diff == 1 || diff == -3 || (chip_id % 2 == 0 && num_hops == 2)) {
                    forward_connection = true;
                } else {
                    forward_connection = false;
                }
                unicast_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
            } else {
                const uint32_t num_hops = std::abs(int(target_device_id) - int(chip_id));
                unicast_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
                forward_connection = target_device_id > chip_id;
            }
            auto& fabric_conn = forward_connection ? fabric_connection.get_forward_connection()
                                                   : fabric_connection.get_backward_connection();

            uint32_t num_pages_sent = 0;
            uint32_t packet = sender_packet_start;
            while (num_pages_sent < sender_total_num_pages) {
                // Determine packet size based on whether it's the last packet
                auto num_pages_left = sender_total_num_pages - num_pages_sent;
                const uint32_t curr_packet_num_pages = std::min(num_pages_per_packet, num_pages_left);
                const uint32_t curr_packet_size_bytes = curr_packet_num_pages * page_size_bytes;

                const uint32_t receiver_core_x = packet_worker_cores[packet][x_index];
                const uint32_t receiver_core_y = packet_worker_cores[packet][y_index];
                const uint64_t noc0_dest_noc_addr = get_noc_addr(receiver_core_x, receiver_core_y, packet_offset, 0);

                cb_wait_front(fabric_sender_cb_id, curr_packet_num_pages);
                const auto sender_l1_addr = get_read_ptr(fabric_sender_cb_id);

                const uint64_t sem_noc_addr =
                    get_noc_addr(receiver_core_x, receiver_core_y, receiver_semaphore_address, 0);
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
        constexpr uint8_t q_output_core_xy[output_cores_per_device][2] = Q_OUTPUT_CORE_XY;
        constexpr uint8_t k_output_core_xy[output_cores_per_device][2] = K_OUTPUT_CORE_XY;
        constexpr uint8_t v_output_core_xy[output_cores_per_device][2] = V_OUTPUT_CORE_XY;

        uint32_t head_idx = linear_output_page_start_idx / 2;  // each head has 2 pages/blocks
        cb_wait_front(accumulator_cb_id, num_pages_per_packet);

        auto accumulator_l1_addr = get_read_ptr(accumulator_cb_id);
        if (head_idx < q_heads) {  // write q heads
            for (uint32_t iblock = 0; iblock < num_pages_per_packet;
                 iblock += 2) {  // increment by 2 so that we can handle 1 head each time.
                for (uint32_t istick = 0; istick < num_sticks_per_block; istick++) {
                    uint64_t noc_address = get_noc_addr(
                        q_output_core_xy[istick][x_index],
                        q_output_core_xy[istick][y_index],
                        q_base_addr + head_idx * head_dim_bytes);
                    uint32_t l1_read_addr = accumulator_l1_addr + iblock * page_size_bytes + istick * stick_size_byte;
                    noc_async_write(l1_read_addr, noc_address, stick_size_byte);
                }
                head_idx++;  // next head as each packet has 2 heads = 4 blocks
            }
        } else {  // write kv heads
            uint32_t iblock1 = 0, iblock2 = 2;
            for (uint32_t istick = 0; istick < num_sticks_per_block; istick++) {
                uint64_t noc_address =
                    get_noc_addr(k_output_core_xy[istick][x_index], k_output_core_xy[istick][y_index], k_base_addr);
                uint32_t l1_read_addr = accumulator_l1_addr + iblock1 * page_size_bytes + istick * stick_size_byte;
                noc_async_write(l1_read_addr, noc_address, stick_size_byte);
            }

            for (uint32_t istick = 0; istick < num_sticks_per_block; istick++) {
                uint64_t noc_address =
                    get_noc_addr(v_output_core_xy[istick][x_index], v_output_core_xy[istick][y_index], v_base_addr);
                uint32_t l1_read_addr = accumulator_l1_addr + iblock2 * page_size_bytes + istick * stick_size_byte;
                noc_async_write(l1_read_addr, noc_address, stick_size_byte);
            }
        }
        noc_async_write_barrier();
    }
}
