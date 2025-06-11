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

enum Direction {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
};

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
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);  // ew_dim

    constexpr bool directions[4] = {
        (bool)get_compile_time_arg_val(32),
        (bool)get_compile_time_arg_val(33),
        (bool)get_compile_time_arg_val(34),
        (bool)get_compile_time_arg_val(35)};

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i]) {
            fabric_connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            fabric_connections[i].open();
        }
    }

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    constexpr uint8_t routers[num_devices] = ROUTE;

    auto output_addr_gen = get_interleaved_addr_gen<output_is_dram, output_page_size>(output_tensor_address);
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

    for (uint32_t local_token = 0; local_token < batches_per_device; local_token++) {
        uint32_t b = local_token + batches_per_device * src_chip_id;
        uint32_t input_token_read_addr = get_read_ptr(input_tensor_cb_id) + local_token * input_page_size;
        uint64_t output_token_write_addr = get_noc_addr(b, output_addr_gen);
        for (uint32_t k = 0; k < selected_experts_k; k++) {
            uint32_t offset = local_token * indices_page_size + k * sizeof(uint16_t);
            uint16_t expert_chosen = *((uint16_t*)get_read_ptr(indices_tensor_cb_id) + offset);
            uint32_t expert_offset = expert_chosen * mapping_page_size;

            uint16_t* devices_for_expert = (uint16_t*)(get_read_ptr(mapping_tensor_cb_id) + expert_offset);

            for (uint32_t d = 0; d < num_devices; d++) {
                if (devices_for_expert[d] == 1) {
                    if (dest_chip_ids[d] == src_chip_id) {
                        // simply write via local noc
                        noc_async_write(input_token_read_addr, output_token_write_addr, input_page_size);
                        noc_async_write_barrier();
                    } else {
                        unicast_packet_header->to_noc_unicast_write(
                            NocUnicastCommandHeader{output_token_write_addr}, input_page_size);
                        uint8_t route = routers[d];
                        uint8_t dest_chip_id = dest_chip_ids[d];
                        uint8_t dest_mesh_id = dest_mesh_ids[d];
                        fabric_set_unicast_route(
                            (LowLatencyMeshPacketHeader*)packet_header_buffer_address,
                            (eth_chan_directions)fabric_connections[route].direction,
                            src_chip_id,
                            dest_chip_id,
                            dest_mesh_id,
                            mesh_cols);

                        send_packet(
                            unicast_packet_header,
                            output_token_write_addr,
                            input_token_read_addr,
                            input_page_size,
                            fabric_connections[route]);
                    }
                }
            }
        }
    }

    // send semaphore increment to all other devices
    for (uint32_t local_token = 0; local_token < batches_per_device; local_token++) {
        uint32_t b = local_token + batches_per_device * src_chip_id;
        for (uint32_t d = 0; d < num_devices; d++) {
            if (dest_chip_ids[d] == src_chip_id) {
                // send metadata to local device output buffer
                uint32_t token_indices_address = get_read_ptr(indices_tensor_cb_id) + local_token * indices_page_size;
                uint64_t metadata_write_addr = get_noc_addr(b, metadata_addr_gen);
                noc_async_write(token_indices_address, metadata_write_addr, indices_page_size);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(global_semaphore_address), 1);
                noc_async_atomic_barrier();
            } else {
                // send metadata to other devices
                uint32_t token_indices_address = get_read_ptr(indices_tensor_cb_id) + local_token * indices_page_size;
                uint64_t metadata_write_addr = get_noc_addr(b, metadata_addr_gen);
                uint64_t global_noc_semaphore_address = get_noc_addr(global_semaphore_address);

                metadata_packet_header->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                        metadata_write_addr, global_noc_semaphore_address, 1, 32, true),
                    indices_page_size);

                fabric_set_unicast_route(
                    (LowLatencyMeshPacketHeader*)metadata_packet_header,
                    (eth_chan_directions)fabric_connections[routers[d]].direction,
                    src_chip_id,
                    dest_chip_ids[d],
                    dest_mesh_ids[d],
                    mesh_cols);

                fabric_connections[routers[d]].wait_for_empty_write_slot();
                fabric_connections[routers[d]].send_payload_without_header_non_blocking_from_address(
                    token_indices_address, indices_page_size);
                fabric_connections[routers[d]].send_payload_flush_blocking_from_address(
                    (uint32_t)metadata_packet_header, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }

    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i]) {
            fabric_connections[i].close();
        }
    }
}
