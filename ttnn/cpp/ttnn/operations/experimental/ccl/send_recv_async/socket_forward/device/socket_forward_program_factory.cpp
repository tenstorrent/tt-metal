// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_forward_program_factory.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::operations::experimental::ccl::socket_forward {

SocketForwardMeshWorkloadFactory::cached_mesh_workload_t SocketForwardMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    const auto& send_socket = operation_attributes.send_socket;
    const auto& recv_socket = operation_attributes.recv_socket;
    const auto& send_socket_connections = send_socket.get_config().socket_connection_config;
    const auto& recv_socket_connections = recv_socket.get_config().socket_connection_config;

    TT_FATAL(send_socket_connections.size() == 1, "SocketForward only supports one sender and one receiver core.");
    TT_FATAL(recv_socket_connections.size() == 1, "SocketForward only supports one sender and one receiver core.");
    TT_FATAL(
        send_socket_connections[0].sender_core.device_coord == recv_socket_connections[0].receiver_core.device_coord,
        "Sender and receiver cores must be on the same device.");

    TT_FATAL(
        send_socket_connections[0].sender_core.core_coord == recv_socket_connections[0].receiver_core.core_coord,
        "Sender and receiver must be on the same core.");

    auto cached_program = create_at(
        operation_attributes, send_socket_connections[0].sender_core.device_coord, tensor_args, tensor_return_value);
    workload.add_program(
        ttnn::MeshCoordinateRange(send_socket_connections[0].sender_core.device_coord),
        std::move(cached_program.program));
    shared_variables.emplace(
        ttnn::MeshCoordinateRange(send_socket_connections[0].sender_core.device_coord),
        std::move(cached_program.shared_variables));
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<SocketForwardSharedVariables> SocketForwardMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    tt::tt_metal::Program program{};

    const auto& send_socket = operation_attributes.send_socket;
    const auto& recv_socket = operation_attributes.recv_socket;
    auto num_bytes = operation_attributes.num_bytes;

    auto mesh_device = send_socket.get_device();
    const auto& send_socket_conn = send_socket.get_config().socket_connection_config[0];
    const auto& recv_socket_conn = recv_socket.get_config().socket_connection_config[0];

    MeshCoordinate my_device_coord = send_socket_conn.sender_core.device_coord;
    MeshCoordinate upstream_device_coord = recv_socket_conn.sender_core.device_coord;
    MeshCoordinate downstream_device_coord = send_socket_conn.receiver_core.device_coord;

    auto my_core_coord = send_socket_conn.sender_core.core_coord;
    auto receiver_core_coord = send_socket_conn.receiver_core.core_coord;

    tt::tt_fabric::FabricNodeId my_fabric_node_id = mesh_device->get_fabric_node_id(my_device_coord);
    tt::tt_fabric::FabricNodeId upstream_fabric_node_id =
        recv_socket.get_fabric_node_id(tt::tt_metal::distributed::SocketEndpoint::SENDER, upstream_device_coord);
    tt::tt_fabric::FabricNodeId downstream_fabric_node_id =
        send_socket.get_fabric_node_id(tt::tt_metal::distributed::SocketEndpoint::RECEIVER, downstream_device_coord);

    auto max_alignment = std::max(
        mesh_device->allocator()->get_alignment(send_socket.get_config().socket_mem_config.socket_storage_type),
        mesh_device->allocator()->get_alignment(recv_socket.get_config().socket_mem_config.socket_storage_type));

    auto socket_aligned_page_size = tt::align(num_bytes, max_alignment);

    auto bwd_link_indices = tt::tt_fabric::get_forwarding_link_indices(my_fabric_node_id, upstream_fabric_node_id);
    auto fwd_link_indices = tt::tt_fabric::get_forwarding_link_indices(my_fabric_node_id, downstream_fabric_node_id);
    TT_FATAL(
        fwd_link_indices.size() > 1, "Single core multi link version of SocketForward only supports multiple links.");
    TT_FATAL(bwd_link_indices.size(), "No link indices found from downstream to upstream core in SocketForward.");
    uint32_t num_fwd_links = 1;
    uint32_t num_bwd_links = 1;

    auto fabric_max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    uint32_t partial_packet_size = num_bytes % fabric_max_payload_size;
    uint32_t socket_block_size = socket_aligned_page_size;

    uint32_t num_whole_packets = num_bytes / fabric_max_payload_size;
    uint32_t num_whole_packets_link_0 =
        (num_whole_packets / num_fwd_links) + static_cast<uint32_t>(partial_packet_size > 0);

    uint32_t packet_header_cb_num_pages = num_fwd_links + num_bwd_links;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, my_core_coord, cb_packet_header_config);

    constexpr uint32_t barrier_address = 1105600;
    std::vector<uint32_t> compile_args = {
        packet_header_cb_index,
        socket_block_size,
        partial_packet_size,
        fabric_max_payload_size,
        num_whole_packets_link_0,
        barrier_address,
    };

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/socket_forward/device/kernels/socket_forward.cpp",
        my_core_coord,
        tt::tt_metal::WriterDataMovementConfig(compile_args));

    // Need to update this to be downstream bank id
    uint32_t downstream_bank_id = mesh_device->allocator()->get_bank_ids_from_logical_core(
        send_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];

    std::vector<uint32_t> rt_args = {
        send_socket.get_config_buffer()->address(), recv_socket.get_config_buffer()->address(), downstream_bank_id};

    for (uint32_t i = 0; i < num_bwd_links; i++) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            my_fabric_node_id, upstream_fabric_node_id, bwd_link_indices[i], program, my_core_coord, rt_args);
    }
    for (uint32_t i = 0; i < num_fwd_links; i++) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            my_fabric_node_id, downstream_fabric_node_id, fwd_link_indices[i], program, my_core_coord, rt_args);
    }

    tt::tt_metal::SetRuntimeArgs(program, kernel_id, my_core_coord, rt_args);

    return {std::move(program), {.core_coord = my_core_coord, .kernel_id = kernel_id}};
}

void SocketForwardMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        auto& core_coord = shared_vars.core_coord;
        auto& kernel_id = shared_vars.kernel_id;
        auto& runtime_args = GetRuntimeArgs(program, kernel_id, core_coord);
        runtime_args[0] = operation_attributes.send_socket.get_config_buffer()->address();
        runtime_args[1] = operation_attributes.recv_socket.get_config_buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::ccl::socket_forward
