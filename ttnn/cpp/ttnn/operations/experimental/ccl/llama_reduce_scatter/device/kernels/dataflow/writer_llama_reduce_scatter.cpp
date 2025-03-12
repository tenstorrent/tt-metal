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
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"

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
    DPRINT << "Starting kernel_main for writer" << ENDL();
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
    constexpr uint32_t output_cores_per_device = get_compile_time_arg_val(13);

    constexpr uint32_t input_tensor_cores = input_shard_cores_per_device * num_devices;

    // Calculate the total number of packets needed to send all the tiles
    constexpr uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;
    constexpr uint32_t last_packet_num_pages =
        (input_shard_cores_per_device * tiles_per_core_width % num_pages_per_packet == 0)
            ? num_pages_per_packet
            : input_shard_cores_per_device * tiles_per_core_width % num_pages_per_packet;

    DPRINT << "num_packets_total_per_device: " << num_packets_total_per_device << ENDL();
    DPRINT << "last_packet_num_pages: " << last_packet_num_pages << ENDL();
    DPRINT << "input_shard_cores_per_device: " << input_shard_cores_per_device << ENDL();
    DPRINT << "tiles_per_core_width: " << tiles_per_core_width << ENDL();
    DPRINT << "num_pages_per_packet: " << num_pages_per_packet << ENDL();
    DPRINT << "num_devices: " << num_devices << ENDL();
    DPRINT << "page_size_bytes: " << page_size_bytes << ENDL();
    DPRINT << "output_cores_per_device: " << output_cores_per_device << ENDL();

    constexpr uint32_t device_order[num_devices - 1] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
    constexpr uint32_t receiver_core_for_device[num_devices][2] = RECEIVER_CORE_XY;
    constexpr uint32_t input_core_xy[input_tensor_cores][2] = INPUT_CORE_XY;
    constexpr uint32_t output_core_xy[output_cores_per_device][2] = OUTPUT_CORE_XY;

    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t local_semaphore_address = get_semaphore(get_arg_val<uint32_t>(rt_arg_idx++));
    bool sender_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    bool receiver_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t receiver_for_device_id = get_arg_val<uint32_t>(rt_arg_idx++);

    DPRINT << "receiver_semaphore_address " << receiver_semaphore_address << " local_semaphore_address "
           << local_semaphore_address << ENDL();
    DPRINT << "receiver_semaphore_address value: " << *(uint32_t*)receiver_semaphore_address
           << " local_semaphore_address value: " << *(uint32_t*)local_semaphore_address << ENDL();
    DPRINT << "Is receiver_core: " << (uint32_t)receiver_core << " Is sender_core: " << (uint32_t)sender_core << ENDL();

    if (sender_core) {
        auto packet_header_buffer_addr = get_write_ptr(packet_header_cb_id);
        auto* unicast_packet_header = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
        auto* sem_inc_packet_header =
            reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr + sizeof(PACKET_HEADER_TYPE));
        DPRINT << "Opening fabric connection" << ENDL();
        auto fabric_connection = FabricConnectionManager::build_from_args(rt_arg_idx);
        if (fabric_connection.is_logically_connected()) {
            fabric_connection.open();
        }
        DPRINT << "Fabric connection opened " << ENDL();
        for (auto target_device_id : device_order) {
            if (target_device_id == chip_id) {
                break;
            }
            uint32_t num_hops = std::abs(int(target_device_id) - int(chip_id));
            unicast_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
            auto& fabric_conn = target_device_id > chip_id ? fabric_connection.get_forward_connection()
                                                           : fabric_connection.get_backward_connection();

            uint32_t receiver_core_x = receiver_core_for_device[target_device_id][0];
            uint32_t receiver_core_y = receiver_core_for_device[target_device_id][1];

            // for LLaMa - 6 cores * 5 tiles per core = 30 tiles to each other device
            // 30/4 = 8 packets, with the last packet having 2 pages
            uint32_t packet_offset = target_device_id * input_shard_cores_per_device * tiles_per_core_width;
            uint32_t base_receiver_l1_addr = get_read_ptr(fabric_receiver_cb_id);
            for (uint32_t packet = 0; packet < num_packets_total_per_device; packet++) {
                uint32_t curr_packet_num_pages =
                    packet == num_packets_total_per_device - 1 ? last_packet_num_pages : num_pages_per_packet;
                DPRINT << "curr_packet_num_pages: " << curr_packet_num_pages << ENDL();
                cb_wait_front(fabric_sender_cb_id, curr_packet_num_pages);
                DPRINT << "cb_wait_front done" << ENDL();
                auto sender_l1_addr = get_write_ptr(fabric_sender_cb_id);
                print_tiles(fabric_sender_cb_id, 0, curr_packet_num_pages, true);

                uint64_t noc0_dest_noc_addr =
                    get_noc_addr(receiver_core_x, receiver_core_y, base_receiver_l1_addr + packet_offset);

                unicast_packet_header->to_noc_unicast_write(
                    tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr},
                    curr_packet_num_pages * page_size_bytes);

                fabric_conn.wait_for_empty_write_slot();
                fabric_conn.send_payload_without_header_non_blocking_from_address(
                    sender_l1_addr, curr_packet_num_pages * page_size_bytes);
                fabric_conn.send_payload_flush_blocking_from_address(
                    (uint32_t)unicast_packet_header, sizeof(PACKET_HEADER_TYPE));
                noc_async_writes_flushed();  // flushed because cross chip?

                cb_pop_front(fabric_sender_cb_id, curr_packet_num_pages);
                packet_offset += curr_packet_num_pages;
            }
            DPRINT << "Finished sending packets" << ENDL();

            // 2. mcast to current core 6,6 for now
            uint64_t sem_noc_addr = safe_get_noc_addr(receiver_core_x, receiver_core_y, receiver_semaphore_address, 0);

            sem_inc_packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                sem_noc_addr,
                static_cast<uint16_t>(1),  // increment 1
                32});
            // Write the mcast packet (forward)

            sem_inc_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
            fabric_conn.wait_for_empty_write_slot();
            fabric_conn.send_payload_flush_blocking_from_address(
                (uint32_t)sem_inc_packet_header, sizeof(PACKET_HEADER_TYPE));
            if (fabric_connection.is_logically_connected()) {
                fabric_connection.close();
            }
            DPRINT << "Closing fabric connection" << ENDL();
        }
    } else if (receiver_core) {
        DPRINT << "Receiver for device id: " << receiver_for_device_id << " chip_id: " << chip_id << ENDL();
        if (receiver_for_device_id == chip_id) {
            // If I'm the receiver for the current device, I don't need to do anything as all the data is already here
            return;
        }
        uint32_t base_receiver_l1_addr = get_read_ptr(fabric_receiver_cb_id);

        uint32_t curr_output_core = 0;
        while (*(uint32_t*)receiver_semaphore_address < 1) {
            // Wait for the semaphore to be set
            DPRINT << "Waiting for semaphore to be set" << ENDL();
        }
        print_tiles(fabric_receiver_cb_id, 0, input_shard_cores_per_device * tiles_per_core_width, true);

        uint32_t output_tile_offset = receiver_for_device_id * tiles_per_core_width_output * page_size_bytes;
        uint32_t accumulator_l1_addr = get_read_ptr(accumulator_cb_id);

        for (uint32_t tile = 0; tile < input_shard_cores_per_device * tiles_per_core_width; tile++) {
            // one tile to each core
            uint32_t output_core = tile;
            uint32_t output_core_x = output_core_xy[output_core][0];
            uint32_t output_core_y = output_core_xy[output_core][1];
            uint32_t noc_accumulator_addr =
                get_noc_addr(output_core_x, output_core_y, accumulator_l1_addr + output_tile_offset);
            uint64_t local_receiver_semaphore_noc_addr =
                get_noc_addr(output_core_x, output_core_y, local_semaphore_address);
            noc_async_write(base_receiver_l1_addr + tile * page_size_bytes, noc_accumulator_addr, page_size_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(local_receiver_semaphore_noc_addr, 1);  // mcast inc is needed, this will tank latency
        }
        *(uint32_t*)receiver_semaphore_address = 0;
    } else {
        // Do nothing
        // win
    }
}
