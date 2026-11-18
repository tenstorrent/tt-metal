// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

namespace detail {

template <typename Src, typename Dst>
inline void dispatch_input_local_device_flushed(
    const Noc& noc,
    const Src& src,
    const Dst& dst,
    uint32_t size_bytes,
    const typename noc_traits_t<Src>::src_args_type& src_args,
    const typename noc_traits_t<Dst>::dst_args_type& dst_args) {
    noc.async_write(src, dst, size_bytes, src_args, dst_args);
    noc.async_writes_flushed();
}

// Insert helper that handles the local-device metadata path
template <typename MetadataAddrGen>
inline void dispatch_metadata_local_device(
    const Noc& noc,
    uint32_t token_indices_address,
    const MetadataAddrGen& metadata_addr_gen,
    uint32_t metadata_page_id,
    uint32_t metadata_page_size,
    uint64_t global_noc_semaphore_address,
    bool increment) {
    // send metadata to local device output buffer
    noc.async_write(
        CoreLocalMem<uint32_t>(token_indices_address),
        metadata_addr_gen,
        metadata_page_size,
        {},
        {.page_id = metadata_page_id});
    noc.async_write_barrier();
    if (increment) {
        noc_semaphore_inc(global_noc_semaphore_address, 1);
        noc.async_atomic_barrier();
    }
}

void zero_buffer_async(const DataflowBuffer& dfb, uint32_t bytes) {
    Noc noc;
    noc.async_write_zeros(dfb, bytes);
}

void zero_buffer_barrier() {
    Noc noc;
    noc.write_zeros_l1_barrier();
}

}  // namespace detail

using namespace ttnn::operations::ccl::common;

// Sends one completion credit to every device over both arcs, dispatch_devices in all: one device (the antipode on
// an even ring) is in both arcs' range. So every receiver counts dispatch_devices, whatever the ring size.
// A template so non-Ring builds never instantiate the multicast, which static_asserts a 1D axis.
template <
    bool Enable,
    uint32_t LinearizedMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ReplicateGroup Axis>
inline void send_ring_completion_credit(
    const Noc& noc,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header_pos,
    volatile PACKET_HEADER_TYPE* packet_header_neg,
    uint64_t global_noc_semaphore_address) {
    if constexpr (Enable) {
        fabric_multicast_bidirectional_atomic_inc_1d<
            LinearizedMeshCoord,
            Topology,
            MeshRows,
            MeshCols,
            Axis,
            /*DoubleAntipodalAtomicInc=*/true>(
            fabric_connections, packet_header_pos, packet_header_neg, global_noc_semaphore_address);
        noc.async_atomic_barrier();
    }
}

void kernel_main() {
    constexpr uint32_t input_pages = get_arg(args::input_pages);
    constexpr uint32_t indices_pages = get_arg(args::indices_pages);
    constexpr uint32_t mapping_pages = get_arg(args::mapping_pages);
    constexpr uint32_t output_pages = get_arg(args::output_pages);
    constexpr uint32_t metadata_pages = get_arg(args::metadata_pages);

    constexpr uint32_t input_page_size = get_arg(args::input_page_size);
    constexpr uint32_t indices_page_size = get_arg(args::indices_page_size);
    constexpr uint32_t mapping_page_size = get_arg(args::mapping_page_size);
    constexpr uint32_t output_page_size = get_arg(args::output_page_size);
    constexpr uint32_t metadata_page_size = get_arg(args::metadata_page_size);

    constexpr uint32_t num_devices = get_arg(args::num_devices);
    constexpr uint32_t hidden_size = get_arg(args::hidden_size);
    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t selected_experts_k = get_arg(args::selected_experts_k);
    constexpr uint32_t experts = get_arg(args::experts);
    constexpr uint32_t tokens_per_device = get_arg(args::tokens_per_device);

    constexpr uint32_t num_links = get_arg(args::num_links);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_arg(args::topology);

    constexpr uint32_t src_mesh_id = get_arg(args::src_mesh_id);
    constexpr uint32_t src_chip_id = get_arg(args::src_chip_id);
    constexpr uint32_t mesh_rows = get_arg(args::mesh_rows);
    constexpr uint32_t mesh_cols = get_arg(args::mesh_cols);  // ew_dim
    constexpr uint32_t aligned_input_page_size = get_arg(args::aligned_input_page_size);
    constexpr uint32_t aligned_indices_page_size = get_arg(args::aligned_indices_page_size);
    constexpr uint32_t aligned_mapping_page_size = get_arg(args::aligned_mapping_page_size);
    constexpr uint32_t aligned_output_page_size = get_arg(args::aligned_output_page_size);
    constexpr uint32_t aligned_metadata_page_size = get_arg(args::aligned_metadata_page_size);

    constexpr uint32_t fabric_max_packet_size = get_arg(args::fabric_max_packet_size);
    constexpr uint32_t alignment = get_arg(args::alignment);
    constexpr uint32_t linearized_mesh_coord = get_arg(args::linearized_mesh_coord);

    experimental::GlobalSemaphore global_semaphore(global_sem::global_semaphore);
    experimental::GlobalSemaphore init_semaphore(global_sem::init_semaphore);
    uint32_t token_start_idx = get_arg(args::token_start_idx);
    uint32_t token_end_idx = get_arg(args::token_end_idx);

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;

    constexpr uint32_t num_directions = 4;
    // Fabric connections in {East, West, North, South} order; a direction without a neighbor has no connection.
    const std::array<FabricConnectionBindingToken, num_directions> fabric_bindings = {
        fabric::east, fabric::west, fabric::north, fabric::south};
    std::array<bool, num_directions> directions;
    for (uint32_t i = 0; i < num_directions; i++) {
        directions[i] = tt::tt_fabric::has_fabric_connection(fabric_bindings[i]);
    }

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_directions> fabric_connections;
    for (uint32_t i = 0; i < num_directions; i++) {
        if (directions[i]) {
            fabric_connections[i] = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric_bindings[i]);
            fabric_connections[i].open_start();
        }
    }

    Noc noc;
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_indices(dfb::indices);
    DataflowBuffer dfb_mapping(dfb::mapping);
    DataflowBuffer dfb_packet_header(dfb::packet_header);
    DataflowBuffer dfb_send_preparation_buffer(dfb::send_preparation_buffer);

    uint32_t send_preparation_buffer_address = dfb_send_preparation_buffer.get_write_ptr();
    detail::zero_buffer_async(
        dfb_send_preparation_buffer, (token_end_idx - token_start_idx) * num_devices * sizeof(uint8_t));

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

    const auto output_addr_gen = TensorAccessor(tensor::output);
    const auto metadata_addr_gen = TensorAccessor(tensor::metadata);

    uint32_t packet_header_buffer_address = dfb_packet_header.get_read_ptr();
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    auto* completion_neg_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + 2 * sizeof(PACKET_HEADER_TYPE));

    uint32_t base_indices_addr = dfb_indices.get_read_ptr();

    detail::zero_buffer_barrier();
    open_direction_connections_barrier(directions, fabric_connections);

    // Send initialization semaphore to configured targets for synchronization
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore.address());
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
    init_semaphore.wait(dispatch_devices - 1);
    init_semaphore.set(0);

    // Based on the selected experts, we dispatch the input tokens to the corresponding devices
    dfb_mapping.wait_front(mapping_pages);
    uint8_t* send_preparation_buffer = (uint8_t*)send_preparation_buffer_address;
    for (uint32_t local_token = token_start_idx; local_token < token_end_idx; local_token++) {
        // global_token is the global token index for the current token
        // we need the global token index to write to the output buffer – each global token that could potentially be
        // sent has a unique output buffer address to ensure that it is not overwritten by another token
        uint32_t global_token = (local_token + (tokens_per_device * dispatch_index));
        dfb_indices.wait_front(1);
        dfb_input.wait_front(1);
        uint32_t input_token_read_addr = dfb_input.get_read_ptr();
        uint16_t* token_indices = (uint16_t*)(dfb_indices.get_read_ptr());

        for (uint32_t k = 0; k < selected_experts_k; k++) {
            // get the expert that is chosen for the current token
            uint16_t expert_chosen = token_indices[k];
            uint32_t expert_offset = expert_chosen * aligned_mapping_page_size;
            uint16_t* devices_for_expert = (uint16_t*)(dfb_mapping.get_read_ptr() + expert_offset);

            // find the devices that the expert lives on and dispatch the input tokens to them
            // if there is no tensor parallelism, then the token will only be sent to one device
            for (uint32_t d = device_begin_idx; d < device_end_idx; d += device_stride) {
                if (devices_for_expert[d] == 1 &&
                    send_preparation_buffer[(local_token - token_start_idx) * num_devices + d] == 0) {
                    send_preparation_buffer[(local_token - token_start_idx) * num_devices + d] = 1;
                    if (d == linearized_mesh_coord) {
                        // if the expert lives on the current device, we dispatch the input token to it
                        detail::dispatch_input_local_device_flushed(
                            noc,
                            dfb_input,
                            output_addr_gen,
                            output_page_size,
                            {.offset_bytes = 0},
                            {.page_id = global_token, .offset_bytes = 0});
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
        dfb_indices.pop_front(1);
        dfb_input.pop_front(1);
    }
    if (needs_barrier) {
        noc.async_write_barrier();
    }
    // Send our selected experts tensor to all other devices and signal that we are done dispatching the input tokens
    // with a semaphore
    uint64_t global_noc_semaphore_address = get_noc_addr(global_semaphore.address());

    // Ring: payloads to the antipode alternate arcs (routing tie), so a credit on the metadata packet could
    // overtake a payload on the other arc. There metadata carries no credit; a completion credit is sent on
    // both arcs after all payloads instead.
    constexpr bool ring_completion =
        is_1d_topology<topology>() && topology == tt::tt_fabric::Topology::Ring && axis != ReplicateGroup::NONE;
    constexpr uint32_t metadata_increment = ring_completion ? 0 : 1;

    // two modes: send pages directly to the output buffer or send pages to the intermediate buffer and then write to
    // the output buffer latter is slower but is less L1 intensive
#ifndef FULL_PACKET
    {
        for (uint32_t local_token = token_start_idx; local_token < token_end_idx; local_token++) {
            uint32_t global_token = (local_token + (tokens_per_device * dispatch_index));
            uint32_t token_indices_address =
                base_indices_addr + ((local_token - token_start_idx) * aligned_indices_page_size);

            // dispatch the metadata to all other devices
            for (uint32_t d = device_begin_idx; d < device_end_idx; d += device_stride) {
                if (d == linearized_mesh_coord) {
                    // dispatch the metadata to the current device and increment the local copy of the semaphore
                    detail::dispatch_metadata_local_device(
                        noc,
                        token_indices_address,
                        metadata_addr_gen,
                        global_token,
                        metadata_page_size,
                        global_noc_semaphore_address,
                        !ring_completion);
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
                            metadata_increment,
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
                            metadata_increment,
                            true);
                    }
                }
            }
        }
        if constexpr (ring_completion) {
            // Local credit, once the local writes have landed.
            noc.async_write_barrier();
            noc_semaphore_inc(global_noc_semaphore_address, 1);
            noc.async_atomic_barrier();
        }
    }
#else
    {
        uint32_t indices_size = aligned_indices_page_size * tokens_per_device;
        uint32_t indices_size_per_core = aligned_indices_page_size * (token_end_idx - token_start_idx);

        DataflowBuffer dfb_metadata_buffer(dfb::metadata_buffer);
        const uint32_t core_offset_md_write_offset =
            (dispatch_index * indices_size) + (token_start_idx * aligned_indices_page_size);
        uint64_t intermediate_metadata_write_addr = get_noc_addr(dfb_metadata_buffer.get_read_ptr());
        uint64_t noc_core_offset_md_write_addr = intermediate_metadata_write_addr + core_offset_md_write_offset;

        for (uint32_t d = device_begin_idx; d < device_end_idx; d += device_stride) {
            if (d == linearized_mesh_coord) {
                // dispatch the metadata to the local device
                detail::dispatch_input_local_device_flushed(
                    noc,
                    CoreLocalMem<uint32_t>(base_indices_addr),
                    CoreLocalMem<uint32_t>(dfb_metadata_buffer.get_read_ptr()),
                    indices_size_per_core,
                    {},
                    {.offset_bytes = core_offset_md_write_offset});
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
                        metadata_increment,
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
                        metadata_increment,
                        true);
                }
            }
        }
        // Need to wait for the local flushed metadata write.
        noc.async_write_barrier();
        noc_semaphore_inc(global_noc_semaphore_address, 1);
        noc.async_atomic_barrier();
    }
#endif

    send_ring_completion_credit<ring_completion, linearized_mesh_coord, topology, mesh_rows, mesh_cols, axis>(
        noc, fabric_connections, metadata_packet_header, completion_neg_packet_header, global_noc_semaphore_address);

    dfb_mapping.pop_front(mapping_pages);

    close_direction_connections(directions, fabric_connections);
}
