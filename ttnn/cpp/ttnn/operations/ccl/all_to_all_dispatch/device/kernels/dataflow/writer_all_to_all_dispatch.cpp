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

template <uint32_t mesh_cols, uint32_t mesh_rows, int axis>
bool is_configured_target(uint32_t src_chip_id, uint32_t dest_chip_id) {
    // axis is the direction along which we are allowed to send packets
    // axis = 1; means we are allowed to send packets in the row direction
    // axis = 0; means we are allowed to send packets in the column direction
    // axis = -1; means we are allowed to send packets in all directions
    if (axis == 0) {  // check if they're on the same column
        return src_chip_id % mesh_cols == dest_chip_id % mesh_cols;
    } else if (axis == 1) {  // check if they're on the same row
        return src_chip_id / mesh_cols == dest_chip_id / mesh_cols;
    } else {
        return true;  // if axis is not configured, we assume the target is configured, which is the default case, which
                      // is all directions
    }
}

/*
enum eth_chan_directions {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
    COUNT = 4,
};*/

inline eth_chan_directions get_direction(
    uint32_t src_chip_id, uint32_t dest_chip_id, uint32_t mesh_cols, uint32_t mesh_rows) {
    // if along the same row, we go east or west
    if (src_chip_id / mesh_cols == dest_chip_id / mesh_cols) {
        return src_chip_id < dest_chip_id ? eth_chan_directions::EAST : eth_chan_directions::WEST;
    }
    // if along the same column, we go north or south
    else if (src_chip_id % mesh_cols == dest_chip_id % mesh_cols) {
        return src_chip_id < dest_chip_id ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
    }
    // if not along the same row or column, we go north or south; north if dest_chip_id is smaller than src_chip_id
    return dest_chip_id < src_chip_id ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
}

// Insert helper that handles the local-device metadata path
inline void dispatch_metadata_local_device(
    uint32_t token_indices_address,
    uint64_t metadata_write_addr,
    uint32_t metadata_page_size,
    uint64_t global_noc_semaphore_address) {
    // send metadata to local device output buffer
    noc_async_write(token_indices_address, metadata_write_addr, metadata_page_size);
    noc_async_write_barrier();
    noc_semaphore_inc(global_noc_semaphore_address, 1);
    noc_async_atomic_barrier();
}

// Debug-only helper that reads back metadata from L1 and prints its contents
template <typename MetadataAddrGenT>
inline void debug_print_local_metadata(
    uint32_t send_preparation_buffer_cb_id,
    uint32_t batch_idx,
    MetadataAddrGenT& metadata_addr_gen,
    uint32_t selected_experts_k) {
    uint32_t metaread_cb_l1_addr = get_read_ptr(send_preparation_buffer_cb_id);
    noc_async_read_page(batch_idx, metadata_addr_gen, metaread_cb_l1_addr);
    noc_async_read_barrier();
    DPRINT << "Metadata read from L1" << ENDL();
    uint16_t* metadata_read_ptr = reinterpret_cast<uint16_t*>(metaread_cb_l1_addr);
    for (uint32_t i = 0; i < selected_experts_k; i++) {
        DPRINT << "Expert " << i << " is " << metadata_read_ptr[i] << ENDL();
    }
    DPRINT << "Metadata read from L1 completed" << ENDL();
}

// Insert helper that handles the remote-device metadata path with fused atomic increment
inline void dispatch_metadata_remote_device(
    uint32_t src_chip_id,
    uint32_t dest_chip_id,
    uint32_t dest_mesh_id,
    uint32_t mesh_cols,
    uint32_t mesh_rows,
    uint32_t token_indices_address,
    uint64_t metadata_write_addr,
    uint32_t metadata_page_size,
    uint64_t global_noc_semaphore_address,
    volatile PACKET_HEADER_TYPE* metadata_packet_header,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections) {
    // Clear the header buffer region.
    zero_l1_buf((uint32_t*)metadata_packet_header, sizeof(PACKET_HEADER_TYPE));

    uint32_t route = static_cast<uint32_t>(get_direction(src_chip_id, dest_chip_id, mesh_cols, mesh_rows));

    // Populate packet header with routing information
    fabric_set_unicast_route(
        (LowLatencyMeshPacketHeader*)metadata_packet_header,
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        src_chip_id,
        dest_chip_id,
        dest_mesh_id,
        mesh_cols);

    // Fill header for fused unicast + atomic increment command
    metadata_packet_header->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
            metadata_write_addr, global_noc_semaphore_address, 1, 32, true),
        metadata_page_size);

    // Send payload followed by header over the fabric.
    fabric_connections[route].wait_for_empty_write_slot();

    fabric_connections[route].send_payload_without_header_non_blocking_from_address(
        token_indices_address, metadata_page_size);

    fabric_connections[route].send_payload_flush_blocking_from_address(
        (uint32_t)metadata_packet_header, sizeof(PACKET_HEADER_TYPE));
}

inline void dispatch_input_local_device(
    uint32_t input_token_read_addr, uint64_t output_token_write_addr, uint32_t output_page_size) {
    noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
    noc_async_write_barrier();
}

inline void dispatch_input_remote_device(
    uint32_t src_chip_id,
    uint32_t dest_chip_id,
    uint32_t dest_mesh_id,
    uint32_t mesh_cols,
    uint32_t mesh_rows,
    uint32_t input_token_read_addr,
    uint64_t output_token_write_addr,
    uint32_t size,
    uint32_t fabric_max_packet_size,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* token_unicast_packet_header) {
    // Clear the header buffer region.
    uint32_t route = static_cast<uint32_t>(get_direction(src_chip_id, dest_chip_id, mesh_cols, mesh_rows));

    // Populate packet header with routing information
    zero_l1_buf((uint32_t*)token_unicast_packet_header, sizeof(PACKET_HEADER_TYPE));
    fabric_set_unicast_route(
        (LowLatencyMeshPacketHeader*)token_unicast_packet_header,
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        src_chip_id,
        dest_chip_id,
        dest_mesh_id,
        mesh_cols);
    while (size > 0) {
        uint32_t curr_packet_size = std::min(fabric_max_packet_size, size);

        token_unicast_packet_header->to_noc_unicast_write(
            NocUnicastCommandHeader{output_token_write_addr}, curr_packet_size);

        fabric_connections[route].wait_for_empty_write_slot();

        fabric_connections[route].send_payload_without_header_non_blocking_from_address(
            input_token_read_addr, curr_packet_size);

        fabric_connections[route].send_payload_flush_blocking_from_address(
            (uint32_t)token_unicast_packet_header, sizeof(PACKET_HEADER_TYPE));

        input_token_read_addr += curr_packet_size;
        output_token_write_addr += curr_packet_size;
        size -= curr_packet_size;
    }
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
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(25);

    constexpr uint32_t num_links = get_compile_time_arg_val(26);
    constexpr bool is_ring_topology = (bool)get_compile_time_arg_val(27);

    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(28);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(29);
    // if (!(src_chip_id == 1 || src_chip_id == 5)) {
    //     return;
    // }
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);  // ew_dim
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(32);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_mapping_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(36);

    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(37);

    DPRINT << "Kernel started" << ENDL();

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

#ifdef AXIS
    constexpr int axis = AXIS;
    constexpr uint32_t dispatch_devices = axis == 0 ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index = axis == 0 ? src_chip_id % mesh_rows : src_chip_id % mesh_cols;
#else
    constexpr int axis = -1;
    constexpr uint32_t dispatch_devices = num_devices;
    constexpr uint32_t dispatch_index = src_chip_id;
#endif

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i] == true) {
            fabric_connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            fabric_connections[i].open();
        }
    }

    auto output_addr_gen = get_interleaved_addr_gen<output_is_dram, output_page_size>(output_tensor_address);
    auto metadata_addr_gen = get_interleaved_addr_gen<metadata_is_dram, metadata_page_size>(metadata_tensor_address);

    uint32_t packet_header_buffer_address = get_read_ptr(packet_header_cb_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    cb_wait_front(input_tensor_cb_id, input_pages);
    cb_wait_front(indices_tensor_cb_id, indices_pages);
    cb_wait_front(mapping_tensor_cb_id, mapping_pages);

    for (uint32_t local_token = 0; local_token < tokens_per_device; local_token++) {
        uint32_t b = (local_token + (tokens_per_device * dispatch_index));
        uint32_t input_token_read_addr = get_read_ptr(input_tensor_cb_id) + local_token * aligned_input_page_size;
        uint16_t* token_indices =
            (uint16_t*)(get_read_ptr(indices_tensor_cb_id) + (local_token * aligned_indices_page_size));
        uint64_t output_token_write_addr = get_noc_addr(b, output_addr_gen);
        for (uint32_t k = 0; k < selected_experts_k; k++) {
            uint16_t expert_chosen = token_indices[k];
            uint32_t expert_offset = expert_chosen * aligned_mapping_page_size;
            uint16_t* devices_for_expert = (uint16_t*)(get_read_ptr(mapping_tensor_cb_id) + expert_offset);
            for (uint32_t d = 0; d < num_devices; d++) {
                if (devices_for_expert[d] == 1) {
                    if (dest_chip_ids[d] == src_chip_id) {
                        // simply write via local noc
                        dispatch_input_local_device(input_token_read_addr, output_token_write_addr, output_page_size);
                    } else if (is_configured_target<mesh_cols, mesh_rows, axis>(src_chip_id, dest_chip_ids[d])) {
                        dispatch_input_remote_device(
                            src_chip_id,
                            dest_chip_ids[d],
                            dest_mesh_ids[d],
                            mesh_cols,
                            mesh_rows,
                            input_token_read_addr,
                            output_token_write_addr,
                            output_page_size,
                            fabric_max_packet_size,
                            fabric_connections,
                            unicast_packet_header);
                    }
                }
            }
        }
    }

    // send semaphore increment to all other devices

    uint64_t global_noc_semaphore_address = get_noc_addr(global_semaphore_address);
    for (uint32_t local_token = 0; local_token < tokens_per_device; local_token++) {
        uint32_t b = (local_token + (tokens_per_device * dispatch_index));
        uint64_t metadata_write_addr = get_noc_addr(b, metadata_addr_gen);
        uint32_t token_indices_address = get_read_ptr(indices_tensor_cb_id) + (local_token * aligned_indices_page_size);
        for (uint32_t d = 0; d < num_devices; d++) {
            if (dest_chip_ids[d] == src_chip_id) {
                dispatch_metadata_local_device(
                    token_indices_address, metadata_write_addr, metadata_page_size, global_noc_semaphore_address);

                // debug_print_local_metadata(send_preparation_buffer_cb_id, b, metadata_addr_gen, selected_experts_k);

            } else if (is_configured_target<mesh_cols, mesh_rows, axis>(src_chip_id, dest_chip_ids[d])) {
                dispatch_metadata_remote_device(
                    src_chip_id,
                    dest_chip_ids[d],
                    dest_mesh_ids[d],
                    mesh_cols,
                    mesh_rows,
                    token_indices_address,
                    metadata_write_addr,
                    metadata_page_size,
                    global_noc_semaphore_address,
                    metadata_packet_header,
                    fabric_connections);
            }
        }
    }

    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i] == true) {
            fabric_connections[i].close();
        }
    }
    DPRINT << "Kernel finished" << ENDL();
}
