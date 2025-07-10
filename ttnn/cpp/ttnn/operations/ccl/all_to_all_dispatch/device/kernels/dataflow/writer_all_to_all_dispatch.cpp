// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

namespace detail {

inline void dispatch_input_local_device(
    uint32_t input_token_read_addr, uint64_t output_token_write_addr, uint32_t output_page_size) {
    noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
    noc_async_write_barrier();
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
    uint32_t route = get_next_hop_router_direction(dest_mesh_id, dest_chip_id);

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
}  // namespace detail

using namespace ttnn::operations::ccl::common;

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
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);  // ew_dim
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(32);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_mapping_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(36);

    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(37);
    constexpr uint32_t alignment = get_compile_time_arg_val(38);

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;

    constexpr uint32_t num_directions = 4;
    constexpr std::array<bool, num_directions> directions = DIRECTIONS;

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_directions> fabric_connections;
    open_direction_connections(directions, fabric_connections, rt_args_idx);

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index =
        axis == ReplicateGroup::COLS ? src_chip_id / mesh_cols : src_chip_id % mesh_cols;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t dispatch_devices = num_devices;
    constexpr uint32_t dispatch_index = src_chip_id;
#endif

    auto output_addr_gen = get_interleaved_addr_gen<output_is_dram, output_page_size>(output_tensor_address);
    auto metadata_addr_gen = get_interleaved_addr_gen<metadata_is_dram, metadata_page_size>(metadata_tensor_address);

    uint32_t packet_header_buffer_address = get_read_ptr(packet_header_cb_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    uint32_t base_indices_addr = get_read_ptr(indices_tensor_cb_id);

    // Based on the selected experts, we dispatch the input tokens to the corresponding devices
    cb_wait_front(mapping_tensor_cb_id, mapping_pages);
    for (uint32_t local_token = 0; local_token < tokens_per_device; local_token++) {
        // global_token is the global token index for the current token
        // we need the global token index to write to the output buffer – each global token that could potentially be
        // sent has a unique output buffer address to ensure that it is not overwritten by another token
        uint32_t global_token = (local_token + (tokens_per_device * dispatch_index));
        uint64_t output_token_write_addr = get_noc_addr(global_token, output_addr_gen);
        cb_wait_front(indices_tensor_cb_id, 1);
        cb_wait_front(input_tensor_cb_id, 1);
        uint32_t input_token_read_addr = get_read_ptr(input_tensor_cb_id);
        uint16_t* token_indices = (uint16_t*)(get_read_ptr(indices_tensor_cb_id));

        for (uint32_t k = 0; k < selected_experts_k; k++) {
            // get the expert that is chosen for the current token
            uint16_t expert_chosen = token_indices[k];
            uint32_t expert_offset = expert_chosen * aligned_mapping_page_size;
            uint16_t* devices_for_expert = (uint16_t*)(get_read_ptr(mapping_tensor_cb_id) + expert_offset);

            // find the devices that the expert lives on and dispatch the input tokens to them
            // if there is no tensor parallelism, then the token will only be sent to one device
            for (uint32_t d = 0; d < num_devices; d++) {
                if (devices_for_expert[d] == 1) {
                    if (dest_chip_ids[d] == src_chip_id) {
                        // if the expert lives on the current device, we dispatch the input token to it
                        detail::dispatch_input_local_device(
                            input_token_read_addr, output_token_write_addr, output_page_size);
                    } else if (is_configured_target<src_chip_id, mesh_cols, mesh_rows, axis>(dest_chip_ids[d])) {
                        // if the expert lives on a remote device, we dispatch the input token to it
                        // if axis is specified then we only send to the devices that are along the axis
                        // if axis is not specified then we send to all devices
                        dispatch_input_remote_device<src_chip_id, mesh_cols, mesh_rows, fabric_max_packet_size>(
                            dest_chip_ids[d],
                            dest_mesh_ids[d],
                            alignment,
                            (int)output_page_size,
                            input_token_read_addr,
                            output_token_write_addr,
                            fabric_connections,
                            unicast_packet_header);
                    }
                }
            }
        }
        cb_pop_front(indices_tensor_cb_id, 1);
        cb_pop_front(input_tensor_cb_id, 1);
    }

    // Send our selected experts tensor to all other devices and signal that we are done dispatching the input tokens
    // with a semaphore
    uint64_t global_noc_semaphore_address = get_noc_addr(global_semaphore_address);
    for (uint32_t local_token = 0; local_token < tokens_per_device; local_token++) {
        uint32_t global_token = (local_token + (tokens_per_device * dispatch_index));
        uint64_t metadata_write_addr = get_noc_addr(global_token, metadata_addr_gen);
        uint32_t token_indices_address = base_indices_addr + (local_token * aligned_indices_page_size);

        // dispatch the metadata to all other devices
        for (uint32_t d = 0; d < num_devices; d++) {
            if (dest_chip_ids[d] == src_chip_id) {
                // dispatch the metadata to the current device and increment the local copy of the semaphore
                detail::dispatch_metadata_local_device(
                    token_indices_address, metadata_write_addr, metadata_page_size, global_noc_semaphore_address);
            } else if (is_configured_target<src_chip_id, mesh_cols, mesh_rows, axis>(dest_chip_ids[d])) {
                // dispatch the metadata to the remote device and increment the remote device's copy of the semaphore
                detail::dispatch_metadata_remote_device(
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
    cb_pop_front(mapping_tensor_cb_id, mapping_pages);

    close_direction_connections(directions, fabric_connections);
}
