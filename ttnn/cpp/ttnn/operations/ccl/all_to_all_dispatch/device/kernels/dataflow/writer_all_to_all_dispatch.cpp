// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

namespace detail {

inline void dispatch_input_local_device_flushed(
    uint32_t input_token_read_addr, uint64_t output_token_write_addr, uint32_t output_page_size) {
    noc_async_write(input_token_read_addr, output_token_write_addr, output_page_size);
    noc_async_writes_flushed();
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

void zero_buffer_async(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
}

void zero_buffer_barrier() { noc_async_read_barrier(); }

}  // namespace detail

using namespace ttnn::operations::ccl::common;

void kernel_main() {
    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t indices_tensor_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t mapping_tensor_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t send_preparation_buffer_cb_id = get_compile_time_arg_val(4);

    constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    constexpr uint32_t mapping_pages = get_compile_time_arg_val(7);
    constexpr uint32_t output_pages = get_compile_time_arg_val(8);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(9);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t mapping_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(14);

    constexpr uint32_t num_devices = get_compile_time_arg_val(15);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(16);
    constexpr uint32_t batch_size = get_compile_time_arg_val(17);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(18);
    constexpr uint32_t experts = get_compile_time_arg_val(19);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(20);

    constexpr uint32_t num_links = get_compile_time_arg_val(21);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(22);

    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(23);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(24);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(25);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(26);  // ew_dim
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(27);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(28);
    constexpr uint32_t aligned_mapping_page_size = get_compile_time_arg_val(29);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(30);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(31);

    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(32);
    constexpr uint32_t alignment = get_compile_time_arg_val(33);
    constexpr uint32_t metadata_buffer_id = get_compile_time_arg_val(34);
    constexpr uint32_t write_page_by_page = get_compile_time_arg_val(35);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(36);

    constexpr auto input_args = TensorAccessorArgs<37>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto mapping_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t global_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;

    constexpr uint32_t num_directions = 4;
    constexpr std::array<bool, num_directions> directions = DIRECTIONS;

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_directions> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);

    uint32_t send_preparation_buffer_address = get_write_ptr(send_preparation_buffer_cb_id);
    detail::zero_buffer_async(
        send_preparation_buffer_address, (token_end_idx - token_start_idx) * num_devices * sizeof(uint8_t));

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;

    constexpr uint32_t dispatch_index = axis == ReplicateGroup::COLS ? row : col;
    // Based on cluster axis, we only need to dispatch to the devices that are along the axis
    // If ReplicateGroup is COLs/AXIS is 1, then we dispatch alonw the ROW, and vice versa
    // For ReplicateGroup COLs/AXIS is 1, the device_begin_idx is the start of the row, and the device_end_idx is the
    // end of the row For ReplicateGroup ROWs/AXIS is 0, the device_begin_idx is the start of the column, and the
    // device_end_idx is the end of the column
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (axis == ReplicateGroup::COLS)
            ? (col + mesh_rows * mesh_cols)   // last is col+(mesh_rows-1)*mesh_cols; add one stride
            : (row * mesh_cols + mesh_cols);  // last is row*mesh_cols+(mesh_cols-1); add one
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t dispatch_devices = num_devices;
    constexpr uint32_t dispatch_index = linearized_mesh_coord;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_end_idx = num_devices;
    constexpr uint32_t device_stride = 1;
#endif

    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, output_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, metadata_page_size);

    uint32_t packet_header_buffer_address = get_read_ptr(packet_header_cb_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    uint32_t base_indices_addr = get_read_ptr(indices_tensor_cb_id);

    detail::zero_buffer_barrier();
    open_direction_connections_barrier(directions, fabric_connections);

    // Send initialization semaphore to configured targets for synchronization
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        num_devices>(fabric_connections, metadata_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    // Wait for all devices to complete initialization synchronization
    bool needs_barrier = false;
    noc_semaphore_wait((uint32_t*)init_semaphore_address, dispatch_devices - 1);
    noc_semaphore_set((uint32_t*)init_semaphore_address, 0);

    // Based on the selected experts, we dispatch the input tokens to the corresponding devices
    cb_wait_front(mapping_tensor_cb_id, mapping_pages);
    uint8_t* send_preparation_buffer = (uint8_t*)send_preparation_buffer_address;
    for (uint32_t local_token = token_start_idx; local_token < token_end_idx; local_token++) {
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
            for (uint32_t d = device_begin_idx; d < device_end_idx; d += device_stride) {
                if (devices_for_expert[d] == 1 &&
                    send_preparation_buffer[(local_token - token_start_idx) * num_devices + d] == 0) {
                    send_preparation_buffer[(local_token - token_start_idx) * num_devices + d] = 1;
                    if (d == linearized_mesh_coord) {
                        // if the expert lives on the current device, we dispatch the input token to it
                        detail::dispatch_input_local_device_flushed(
                            input_token_read_addr, output_token_write_addr, output_page_size);
                        needs_barrier = true;
                    } else if (is_configured_target<linearized_mesh_coord, mesh_rows, mesh_cols, axis>(d)) {
                        // if the expert lives on a remote device, we dispatch the input token to it
                        // if axis is specified then we only send to the devices that are along the axis
                        // if axis is not specified then we send to all devices
                        if constexpr (is_1d_topology<topology>()) {
                            fabric_send_chip_unicast_noc_unicast_1d<
                                linearized_mesh_coord,
                                topology,
                                mesh_rows,
                                mesh_cols,
                                fabric_max_packet_size>(
                                output_addr_gen,
                                fabric_connections,
                                unicast_packet_header,
                                d,
                                input_token_read_addr,
                                global_token,
                                (int)output_page_size,
                                alignment);
                        } else {
                            fabric_send_chip_unicast_noc_unicast<
                                src_chip_id,
                                mesh_rows,
                                mesh_cols,
                                fabric_max_packet_size>(
                                output_addr_gen,
                                fabric_connections,
                                unicast_packet_header,
                                dest_chip_ids[d],
                                dest_mesh_ids[d],
                                input_token_read_addr,
                                global_token,
                                (int)output_page_size,
                                alignment);
                        }
                    }
                }
            }
        }
        cb_pop_front(indices_tensor_cb_id, 1);
        cb_pop_front(input_tensor_cb_id, 1);
    }
    if (needs_barrier) {
        noc_async_write_barrier();
    }
    // Send our selected experts tensor to all other devices and signal that we are done dispatching the input tokens
    // with a semaphore
    uint64_t global_noc_semaphore_address = get_noc_addr(global_semaphore_address);

    // two modes: send pages directly to the output buffer or send pages to the intermediate buffer and then write to
    // the output buffer latter is slower but is less L1 intensive
    if constexpr (write_page_by_page) {
        for (uint32_t local_token = token_start_idx; local_token < token_end_idx; local_token++) {
            uint32_t global_token = (local_token + (tokens_per_device * dispatch_index));
            uint64_t metadata_write_addr = get_noc_addr(global_token, metadata_addr_gen);
            uint32_t token_indices_address =
                base_indices_addr + ((local_token - token_start_idx) * aligned_indices_page_size);

            // dispatch the metadata to all other devices
            for (uint32_t d = device_begin_idx; d < device_end_idx; d += device_stride) {
                if (d == linearized_mesh_coord) {
                    // dispatch the metadata to the current device and increment the local copy of the semaphore
                    detail::dispatch_metadata_local_device(
                        token_indices_address, metadata_write_addr, metadata_page_size, global_noc_semaphore_address);
                } else if (is_configured_target<linearized_mesh_coord, mesh_rows, mesh_cols, axis>(d)) {
                    // dispatch the metadata to the remote device and increment the remote device's copy of the
                    // semaphore
                    if constexpr (is_1d_topology<topology>()) {
                        fabric_send_chip_unicast_noc_unicast_with_semaphore_1d<
                            linearized_mesh_coord,
                            topology,
                            mesh_rows,
                            mesh_cols,
                            fabric_max_packet_size>(
                            metadata_addr_gen,
                            fabric_connections,
                            metadata_packet_header,
                            d,
                            token_indices_address,
                            global_token,
                            global_noc_semaphore_address,
                            (int)metadata_page_size,
                            alignment,
                            1,
                            true);
                    } else {
                        fabric_send_chip_unicast_noc_unicast_with_semaphore<
                            src_chip_id,
                            mesh_rows,
                            mesh_cols,
                            fabric_max_packet_size>(
                            metadata_addr_gen,
                            fabric_connections,
                            metadata_packet_header,
                            dest_chip_ids[d],
                            dest_mesh_ids[d],
                            token_indices_address,
                            global_token,
                            global_noc_semaphore_address,
                            (int)metadata_page_size,
                            alignment,
                            1,
                            true);
                    }
                }
            }
        }
    } else {
        uint32_t indices_size = aligned_indices_page_size * tokens_per_device;
        uint32_t indices_size_per_core = aligned_indices_page_size * (token_end_idx - token_start_idx);

        uint64_t intermediate_metadata_write_addr = get_noc_addr(get_read_ptr(metadata_buffer_id));
        uint64_t noc_core_offset_md_write_addr = intermediate_metadata_write_addr + (dispatch_index * indices_size) +
                                                 (token_start_idx * aligned_indices_page_size);

        for (uint32_t d = device_begin_idx; d < device_end_idx; d += device_stride) {
            if (d == linearized_mesh_coord) {
                // dispatch the metadata to the local device
                detail::dispatch_input_local_device_flushed(
                    base_indices_addr, noc_core_offset_md_write_addr, indices_size_per_core);
            } else if (is_configured_target<linearized_mesh_coord, mesh_rows, mesh_cols, axis>(d)) {
                if constexpr (is_1d_topology<topology>()) {
                    l1_only_fabric_send_chip_unicast_noc_unicast_with_semaphore_1d<
                        linearized_mesh_coord,
                        topology,
                        mesh_rows,
                        mesh_cols,
                        fabric_max_packet_size>(
                        fabric_connections,
                        metadata_packet_header,
                        d,
                        base_indices_addr,
                        noc_core_offset_md_write_addr,
                        global_noc_semaphore_address,
                        (int)indices_size_per_core,
                        alignment,
                        1,
                        true);
                } else {
                    l1_only_fabric_send_chip_unicast_noc_unicast_with_semaphore<
                        src_chip_id,
                        mesh_rows,
                        mesh_cols,
                        fabric_max_packet_size>(
                        fabric_connections,
                        metadata_packet_header,
                        dest_chip_ids[d],
                        dest_mesh_ids[d],
                        base_indices_addr,
                        noc_core_offset_md_write_addr,
                        global_noc_semaphore_address,
                        (int)indices_size_per_core,
                        alignment,
                        1,
                        true);
                }
            }
        }
        // Need to wait for the local flushed metadata write.
        noc_async_write_barrier();
        noc_semaphore_inc(global_noc_semaphore_address, 1);
        noc_async_atomic_barrier();
    }

    cb_pop_front(mapping_tensor_cb_id, mapping_pages);

    close_direction_connections(directions, fabric_connections);
}
