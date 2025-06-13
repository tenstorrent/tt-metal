// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"

inline void send_packet(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint64_t noc_dest_addr,
    uint32_t source_l1_buffer_address,
    uint32_t packet_payload_size_bytes,
    tt::tt_fabric::WorkerToFabricEdmSender& connection) {
    connection.wait_for_empty_write_slot();
    connection.send_payload_without_header_non_blocking_from_address(
        source_l1_buffer_address, packet_payload_size_bytes);
    connection.send_payload_flush_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

/*
enum eth_chan_directions {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
    COUNT = 4,
};*/

inline uint32_t get_direction(uint32_t src_chip_id, uint32_t dest_chip_id, uint32_t mesh_cols, uint32_t mesh_rows) {
    // row 0 to row 1
    if (src_chip_id <= 3 && dest_chip_id > 3) {
        return 3;
    }
    // row 1 to row 0
    else if (src_chip_id > 3 && dest_chip_id <= 3) {
        return 2;
    }
    // go east
    else if (dest_chip_id > src_chip_id) {
        return 0;
    }
    // go west
    else if (dest_chip_id < src_chip_id) {
        return 1;
    }
    return 0;
}

void kernel_main() {
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr bool indices_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr bool mapping_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool metadata_is_dram = (bool)get_compile_time_arg_val(4);

    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t indices_tensor_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t mapping_tensor_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t send_preparation_buffer_cb_id = get_compile_time_arg_val(9);

    constexpr uint32_t input_pages = get_compile_time_arg_val(10);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(11);
    constexpr uint32_t mapping_pages = get_compile_time_arg_val(12);
    constexpr uint32_t output_pages = get_compile_time_arg_val(13);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(14);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t mapping_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(19);

    constexpr uint32_t num_devices = get_compile_time_arg_val(20);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(21);
    constexpr uint32_t batch_size = get_compile_time_arg_val(22);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(23);
    constexpr uint32_t experts = get_compile_time_arg_val(24);
    constexpr uint32_t batches_per_device = get_compile_time_arg_val(25);

    constexpr uint32_t num_links = get_compile_time_arg_val(26);
    constexpr bool is_ring_topology = (bool)get_compile_time_arg_val(27);

    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(28);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(29);
    if (src_chip_id != 1) {
        return;
    }
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);  // ew_dim
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(32);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_mapping_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(36);

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    constexpr std::array<bool, 4> directions = DIRECTIONS;

    for (uint32_t i = 0; i < 4; i++) {
        DPRINT << "Direction " << (uint32_t)i << " is " << (directions[i] ? "true" : "false") << ENDL();
    }

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i] == true) {
            DPRINT << "Attempting to open fabric connection for direction: " << (uint32_t)i << ENDL();
            fabric_connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            fabric_connections[i].open();
            DPRINT << "Fabric connection opened for direction: " << (uint32_t)i
                   << " with direction: " << (uint32_t)((eth_chan_directions)fabric_connections[i].direction) << ENDL();
        }
    }

    auto output_addr_gen = get_interleaved_addr_gen<output_is_dram, output_page_size>(output_tensor_address);
    DPRINT << "Metadata is dram: " << (uint32_t)metadata_is_dram << " with page size: " << metadata_page_size
           << " and address: " << metadata_tensor_address << ENDL();
    auto metadata_addr_gen = get_interleaved_addr_gen<metadata_is_dram, metadata_page_size>(metadata_tensor_address);

    uint32_t packet_header_buffer_address = get_read_ptr(packet_header_cb_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);

    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    /**
     * this exists to add packet coalescing in the future
     * std::array<std::array<bool, batches_per_device>, num_devices> device_to_batch = {false};
     */

    cb_wait_front(input_tensor_cb_id, input_pages);
    cb_wait_front(indices_tensor_cb_id, indices_pages);
    cb_wait_front(mapping_tensor_cb_id, mapping_pages);
    DPRINT << "SUCCESSFULLY WAITED FOR ALL TENSORS" << ENDL();
    // for (uint32_t local_token = 0; local_token < batches_per_device; local_token++) {
    //     uint32_t b = local_token + batches_per_device * src_chip_id;
    //     uint32_t input_token_read_addr = get_read_ptr(input_tensor_cb_id) + local_token * input_page_size;
    //     uint64_t output_token_write_addr = get_noc_addr(b, output_addr_gen);
    //     for (uint32_t k = 0; k < selected_experts_k; k++) {
    //         uint32_t offset = local_token * indices_page_size + k * sizeof(uint16_t);
    //         uint16_t expert_chosen = *((uint16_t*)get_read_ptr(indices_tensor_cb_id) + offset);
    //         uint32_t expert_offset = expert_chosen * mapping_page_size;

    //         uint16_t* devices_for_expert = (uint16_t*)(get_read_ptr(mapping_tensor_cb_id) + expert_offset);

    //         for (uint32_t d = 0; d < num_devices; d++) {
    //             if (devices_for_expert[d] == 1) {
    //                 DPRINT << "local token: " << local_token << " batch: " << b << " expert: " << expert_chosen
    //                        << ENDL();
    //                 DPRINT << "sending from device: " << src_chip_id << " to device: " << (uint32_t)dest_chip_ids[d]
    //                        << ENDL();

    //                 if (dest_chip_ids[d] == src_chip_id) {
    //                     // simply write via local noc
    //                     DPRINT << "Token is being written to local device" << ENDL();
    //                     noc_async_write(input_token_read_addr, output_token_write_addr, input_page_size);
    //                     noc_async_write_barrier();
    //                     DPRINT << "Token is written to local device" << ENDL();
    //                 } else {
    //                     DPRINT << "Token is being written to remote device via fabric over dest_mesh_id: "
    //                            << (uint32_t)dest_mesh_ids[d] << ENDL();
    //                     uint32_t route = get_direction(src_chip_id, dest_chip_ids[d], mesh_cols);
    //                     if (route == 0) {
    //                         DPRINT << "Using router: NORTH" << ENDL();
    //                     } else if (route == 1) {
    //                         DPRINT << "Using router: EAST" << ENDL();
    //                     } else if (route == 2) {
    //                         DPRINT << "Using router: SOUTH" << ENDL();
    //                     } else if (route == 3) {
    //                         DPRINT << "Using router: WEST" << ENDL();
    //                     }
    //                     unicast_packet_header->to_noc_unicast_write(
    //                         NocUnicastCommandHeader{output_token_write_addr}, input_page_size);
    //                     uint8_t dest_chip_id = dest_chip_ids[d];
    //                     uint8_t dest_mesh_id = dest_mesh_ids[d];
    //                     fabric_set_unicast_route(
    //                         (LowLatencyMeshPacketHeader*)packet_header_buffer_address,
    //                         (eth_chan_directions)fabric_connections[route].direction,
    //                         src_chip_id,
    //                         dest_chip_id,
    //                         dest_mesh_id,
    //                         mesh_cols);

    //                     send_packet(
    //                         unicast_packet_header,
    //                         output_token_write_addr,
    //                         input_token_read_addr,
    //                         input_page_size,
    //                         fabric_connections[route]);
    //                     DPRINT << "Token is written to remote device" << ENDL();
    //                 }
    //                 DPRINT << "successfully sent token" << ENDL() << ENDL();
    //             }
    //         }
    //     }
    // }
    // DPRINT << "SUCCESSFULLY SENT ALL INPUTS" << ENDL();

    // send semaphore increment to all other devices

    uint64_t global_noc_semaphore_address = get_noc_addr(global_semaphore_address);
    for (uint32_t local_token = 0; local_token < batches_per_device; local_token++) {
        uint32_t b = local_token + (batches_per_device * src_chip_id);
        uint64_t metadata_write_addr = get_noc_addr(b, metadata_addr_gen);
        uint32_t token_indices_address = get_read_ptr(indices_tensor_cb_id) + (local_token * aligned_indices_page_size);
        uint16_t* indices = reinterpret_cast<uint16_t*>(token_indices_address);
        for (uint32_t i = 0; i < selected_experts_k; i++) {
            DPRINT << "Expert " << i << " is " << indices[i] << ENDL();
        }
        DPRINT << ENDL() << "Dispatching metadata for local token " << local_token << " global batch " << b
               << " on device " << src_chip_id << ENDL();
        for (uint32_t d = 0; d < num_devices; d++) {
            if (dest_chip_ids[d] == src_chip_id) {
                DPRINT << "Sending metadata to local device " << (uint32_t)dest_chip_ids[d] << ENDL();
                // send metadata to local device output buffer

                noc_async_write(token_indices_address, metadata_write_addr, metadata_page_size);
                noc_async_write_barrier();
                noc_semaphore_inc(global_noc_semaphore_address, 1);
                noc_async_atomic_barrier();
                DPRINT << "Metadata sent to local device" << ENDL();

                uint32_t metaread_cb_l1_addr = get_read_ptr(send_preparation_buffer_cb_id);
                noc_async_read_page(b, metadata_addr_gen, metaread_cb_l1_addr);
                noc_async_read_barrier();
                DPRINT << "Metadata read from L1" << ENDL();
                uint16_t* metadata_read_ptr = reinterpret_cast<uint16_t*>(metaread_cb_l1_addr);
                for (uint32_t i = 0; i < selected_experts_k; i++) {
                    DPRINT << "Expert " << i << " is " << metadata_read_ptr[i] << ENDL();
                }
                DPRINT << "Metadata read from L1 completed" << ENDL();

            } else {
                // send metadata to other devices
                DPRINT << "Sending metadata to remote device " << (uint32_t)dest_chip_ids[d] << " with address "
                       << metadata_write_addr << " and page size " << metadata_page_size << ENDL();
                zero_l1_buf((uint32_t*)metadata_packet_header, sizeof(PACKET_HEADER_TYPE));

                uint32_t route = get_direction(src_chip_id, dest_chip_ids[d], mesh_cols);
                DPRINT << "From device " << src_chip_id << " to device " << (uint32_t)dest_chip_ids[d] << " via route "
                       << (uint32_t)route << " using direction " << (uint32_t)fabric_connections[route].direction
                       << ENDL();

                fabric_set_unicast_route(
                    (LowLatencyMeshPacketHeader*)metadata_packet_header,
                    (eth_chan_directions)fabric_connections[route].direction,
                    src_chip_id,
                    dest_chip_ids[d],
                    dest_mesh_ids[d],
                    mesh_cols);
                DPRINT << "Fabric unicast route set" << ENDL();

                // metadata_packet_header->to_noc_unicast_write(
                //     tt::tt_fabric::NocUnicastCommandHeader{metadata_write_addr},
                //     metadata_page_size);
                metadata_packet_header->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                        metadata_write_addr, global_noc_semaphore_address, 1, 32, true),
                    metadata_page_size);

                DPRINT << "Noc unicast write atomic inc set" << ENDL();

                fabric_connections[route].wait_for_empty_write_slot();

                DPRINT << "Empty write slot available for fabric connection" << ENDL();

                fabric_connections[route].send_payload_without_header_non_blocking_from_address(
                    token_indices_address, metadata_page_size);

                DPRINT << "Payload of size " << metadata_page_size << " sent to fabric connection" << ENDL();

                fabric_connections[route].send_payload_flush_blocking_from_address(
                    (uint32_t)metadata_packet_header, sizeof(PACKET_HEADER_TYPE));

                DPRINT << "Packet flushed from L1 " << ENDL();
            }
            DPRINT << ENDL();
        }
        DPRINT << "Successfully dispatched metadata" << ENDL() << ENDL();
    }
    DPRINT << "SUCCESSFULLY SENT ALL METADATA" << ENDL();

    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i] == true) {
            fabric_connections[i].close();
        }
    }
    DPRINT << "SUCCESSFULLY CLOSED ALL FABRIC CONNECTIONS" << ENDL();
}
