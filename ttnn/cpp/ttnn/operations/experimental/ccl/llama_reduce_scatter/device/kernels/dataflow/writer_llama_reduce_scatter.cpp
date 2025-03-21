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
    DPRINT << "===" << tile_id << "===" << ENDL();
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

template <uint8_t noc_ind = noc_index>
FORCE_INLINE std::uint64_t static_noc_multicast_addr(
    std::uint32_t noc_x_start,
    std::uint32_t noc_y_start,
    std::uint32_t noc_x_end,
    std::uint32_t noc_y_end,
    std::uint32_t addr) {
    DPRINT << "Noc index " << (uint32_t)noc_ind << ENDL();
    if constexpr (noc_ind == 0) {
        return get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr);
    } else {
        return get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, addr);
    }
}

void kernel_main() {
    DPRINT << "Starting kernel_main for writer" << ENDL();
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
    constexpr uint32_t packet_receiver_core_x = get_compile_time_arg_val(18);
    constexpr uint32_t packet_receiver_core_y = get_compile_time_arg_val(19);

    // Derived compile-time constants
    constexpr uint32_t input_tensor_cores = input_shard_cores_per_device * num_devices;
    constexpr uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;
    constexpr uint32_t last_packet_num_pages =
        (input_shard_cores_per_device * tiles_per_core_width % num_pages_per_packet == 0)
            ? num_pages_per_packet
            : input_shard_cores_per_device * tiles_per_core_width % num_pages_per_packet;
    constexpr size_t packet_header_size = sizeof(PACKET_HEADER_TYPE);

    // Constants for indexing
    constexpr uint32_t x_index = 0;
    constexpr uint32_t y_index = 1;

    constexpr uint8_t device_order[num_devices - 1] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines
    // constexpr uint8_t input_core_xy[input_tensor_cores][2] = INPUT_CORE_XY;
    constexpr uint8_t output_core_xy[output_cores_per_device][2] = OUTPUT_CORE_XY;
    constexpr uint8_t packet_worker_cores[num_packets_total_per_device][2] = PACKET_WORKER_CORES;

    constexpr uint32_t num_dests = (noc_end_x - noc_start_x + 1) * (noc_end_y - noc_start_y + 1);

    // Runtime arguments
    uint32_t receiver_semaphore_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t local_semaphore_address = get_semaphore(get_arg_val<uint32_t>(rt_arg_idx++));
    bool sender_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    bool worker_core = (bool)get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t linear_output_page_start_idx = get_arg_val<uint32_t>(rt_arg_idx++);

    if (sender_core) {
        DPRINT << "SENDER CORE WRITER" << ENDL();

        // Set up packet headers once
        auto packet_header_buffer_addr = get_read_ptr(packet_header_cb_id);
        auto* unicast_packet_header = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
        auto* sem_inc_packet_header =
            reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr + packet_header_size);

        // DPRINT << "Opening fabric connection" << ENDL();
        auto fabric_connection = FabricConnectionManager::build_from_args(rt_arg_idx);
        if (fabric_connection.is_logically_connected()) {
            fabric_connection.open();
        }
        // DPRINT << "Fabric connection opened " << ENDL();
        // DPRINT << "num_packets_total_per_device " << num_packets_total_per_device << ENDL();

        for (auto target_device_id : device_order) {
            if (target_device_id == chip_id) {
                break;
            }

            // Calculate device-specific constants once per device
            uint32_t num_hops = std::abs(int(target_device_id) - int(chip_id));
            unicast_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
            auto& fabric_conn = target_device_id > chip_id ? fabric_connection.get_forward_connection()
                                                           : fabric_connection.get_backward_connection();

            uint32_t base_receiver_l1_addr = get_read_ptr(fabric_receiver_cb_id);

            // for LLaMa - 6 cores * 5 tiles per core = 30 tiles to each other device
            // 30/4 = 8 packets, with the last packet having 2 pages
            uint32_t packet_offset = chip_id * num_pages_per_packet * page_size_bytes;

            for (uint32_t packet = 0; packet < num_packets_total_per_device; packet++) {
                // Determine packet size based on whether it's the last packet
                uint32_t curr_packet_num_pages =
                    packet == num_packets_total_per_device - 1 ? last_packet_num_pages : num_pages_per_packet;
                uint32_t curr_packet_size_bytes = curr_packet_num_pages * page_size_bytes;

                uint32_t receiver_core_x = packet_worker_cores[packet][x_index];
                uint32_t receiver_core_y = packet_worker_cores[packet][y_index];
                DPRINT << "Sending to core " << receiver_core_x << " " << receiver_core_y << " on packet " << packet
                       << " with offset " << packet_offset << ENDL();

                // DPRINT << "packet " << packet << " waiting on " << curr_packet_num_pages << ENDL();
                cb_wait_front(fabric_sender_cb_id, curr_packet_num_pages);
                auto sender_l1_addr = get_read_ptr(fabric_sender_cb_id);
                // print_tiles(fabric_sender_cb_id, 0, curr_packet_num_pages, true);

                uint64_t noc0_dest_noc_addr =
                    get_noc_addr(receiver_core_x, receiver_core_y, base_receiver_l1_addr + packet_offset);

                unicast_packet_header->to_noc_unicast_write(
                    tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, curr_packet_size_bytes);

                // DPRINT << "waiting for empty write slot" << ENDL();
                fabric_conn.wait_for_empty_write_slot();

                // DPRINT << "sending payload without header non-blocking" << ENDL();
                fabric_conn.send_payload_without_header_non_blocking_from_address(
                    sender_l1_addr, curr_packet_size_bytes);

                // DPRINT << "sending payload flush blocking" << ENDL();
                fabric_conn.send_payload_flush_blocking_from_address(
                    (uint32_t)unicast_packet_header, packet_header_size);

                // DPRINT << "flushed because cross chip" << ENDL();
                noc_async_writes_flushed();  // flushed because cross chip?

                // DPRINT << "popping front" << ENDL();
                cb_pop_front(fabric_sender_cb_id, curr_packet_num_pages);
            }

            DPRINT << "sending mcast packet to " << packet_receiver_core_x << " " << packet_receiver_core_y << ENDL();
            uint64_t sem_noc_addr =
                get_noc_addr(packet_receiver_core_x, packet_receiver_core_y, receiver_semaphore_address);
            sem_inc_packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                sem_noc_addr,
                static_cast<uint16_t>(1),  // increment 1
                32});

            // Write the mcast packet (forward)
            // DPRINT << "sending mcast packet to chip " << target_device_id << ENDL();
            sem_inc_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
            fabric_conn.wait_for_empty_write_slot();

            // DPRINT << "sending mcast packet flush blocking" << ENDL();
            fabric_conn.send_payload_flush_blocking_from_address((uint32_t)sem_inc_packet_header, packet_header_size);

            // DPRINT << "flushed because cross chip" << ENDL();
        }

        if (fabric_connection.is_logically_connected()) {
            // DPRINT << "closing fabric connection" << ENDL();
            fabric_connection.close();
            // DPRINT << "fabric connection closed" << ENDL();
        }
        // DPRINT << "Closing fabric connection" << ENDL();
    } else if (worker_core) {
        DPRINT << "WORKER CORE WRITER" << ENDL();
        // DPRINT << "Receiver for device id: " << receiver_for_device_id << " chip_id: " << chip_id << ENDL();
        uint32_t linear_output_core_idcs[num_pages_per_packet];
        uint32_t linear_output_tile_offsets[num_pages_per_packet];

        for (uint32_t i = 0; i < num_pages_per_packet; i++) {
            uint32_t rem = linear_output_page_start_idx + i;
            linear_output_core_idcs[i] = rem / tiles_per_core_width_output;
            linear_output_tile_offsets[i] = rem % tiles_per_core_width_output;
            DPRINT << "linear_output_core_idcs[" << i << "] " << linear_output_core_idcs[i] << ENDL();
            DPRINT << "linear_output_tile_offsets[" << i << "] " << linear_output_tile_offsets[i] << ENDL();
        }

        uint32_t output_tensor_base_addr = get_read_ptr(output_tensor_cb_id);

        DPRINT << "Waiting for accumulator" << ENDL();
        cb_wait_front(accumulator_cb_id, num_pages_per_packet);
        DPRINT << "Accumulator wait done" << ENDL();
        print_tiles(accumulator_cb_id, 0, num_pages_per_packet, true);

        auto accumulator_l1_addr = get_read_ptr(accumulator_cb_id);
        // Process all tiles
        for (uint32_t tile = 0; tile < num_pages_per_packet; tile++) {
            // DPRINT << "writer receiver processing tile " << tile << ENDL();
            // one tile to each core

            uint32_t output_core = linear_output_core_idcs[tile];
            if (linear_output_core_idcs[tile] >= output_cores_per_device) {
                DPRINT << "output_core " << output_core << " is out of bounds" << ENDL();
                break;
            }
            uint32_t output_core_x = output_core_xy[output_core][x_index];
            uint32_t output_core_y = output_core_xy[output_core][y_index];

            // Compute addresses
            uint64_t noc_accumulator_addr = get_noc_addr(
                output_core_x,
                output_core_y,
                output_tensor_base_addr + linear_output_tile_offsets[tile] * page_size_bytes);
            // uint64_t local_receiver_semaphore_noc_addr =
            //     get_noc_addr(output_core_x, output_core_y, local_semaphore_address);

            // print_full_tile(fabric_receiver_cb_id, tile, true);
            noc_async_write(accumulator_l1_addr + tile * page_size_bytes, noc_accumulator_addr, page_size_bytes);
            // noc_async_write_barrier();
            // noc_semaphore_inc(local_receiver_semaphore_noc_addr, 1);  // mcast inc is needed, this will tank latency
        }
        noc_async_write_barrier();
        // DPRINT << "writer receiver async write barrier done" << ENDL();
        // Now we have the block in the CB address, we can mcast to dests!
        // Reset semaphore
        *(uint32_t*)receiver_semaphore_address = 0;
    } else {
        // Do nothing
        // win
    }
    DPRINT << "Kernel finished" << ENDL();
}
