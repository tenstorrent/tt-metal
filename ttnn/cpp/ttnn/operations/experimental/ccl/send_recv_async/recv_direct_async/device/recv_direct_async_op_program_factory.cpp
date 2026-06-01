// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_direct_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

RecvDirectAsyncMeshWorkloadFactory::cached_mesh_workload_t RecvDirectAsyncMeshWorkloadFactory::create_mesh_workload(
    const RecvDirectAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Tensor& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    ttnn::MeshCoordinateRangeSet workload_coords =
        ttnn::send_recv_utils::get_workload_coords<tt::tt_metal::distributed::SocketEndpoint::RECEIVER>(
            tensor_coords, operation_attributes.mesh_socket);
    for (const auto& coord : workload_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<RecvDirectAsyncMeshWorkloadFactory::shared_variables_t>
RecvDirectAsyncMeshWorkloadFactory::create_at(
    const RecvDirectAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/) {
    auto mesh_socket = operation_attributes.mesh_socket;
    const auto& output_tensor = tensor_args;
    auto* mesh_device = output_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.device();

    tt::tt_metal::Program program{};
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    std::vector<CoreCoord> receiver_core_coords;
    std::vector<tt::tt_fabric::FabricNodeId> sender_fabric_node_ids;
    std::vector<tt::tt_fabric::FabricNodeId> receiver_fabric_node_ids;
    std::vector<uint32_t> connection_indices;

    for (uint32_t i = 0; i < socket_connection_config.size(); ++i) {
        const auto& connection = socket_connection_config[i];
        if (socket_mesh_device->get_device(connection.receiver_core.device_coord)->id() == target_device->id()) {
            receiver_core_coords.push_back(connection.receiver_core.core_coord);
            receiver_fabric_node_ids.push_back(
                output_tensor.device()->get_fabric_node_id(connection.receiver_core.device_coord));
            sender_fabric_node_ids.push_back(mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::SENDER, connection.sender_core.device_coord));
            connection_indices.push_back(i);
        }
    }
    uint32_t num_cores = receiver_core_coords.size();

    // cores must not exceed available fabric links
    if (num_cores > 0) {
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[0];
        const auto& sender_fabric_node_id = sender_fabric_node_ids[0];
        auto available_link_indices =
            tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
        uint32_t num_available_links = available_link_indices.size();

        TT_FATAL(
            num_cores <= num_available_links,
            "Cannot create {} receiver-sender pairs with only {} available fabric links between devices. "
            "Reduce the number of cores per device. "
            "Available links: {}, Requested pairs: {}",
            num_cores,
            num_available_links,
            num_available_links,
            num_cores);
    }

    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        output_tensor.buffer()->alignment());
    auto output_page_size = output_tensor.buffer()->aligned_page_size();
    auto total_num_pages = output_tensor.buffer()->num_pages();

    // Must match the sender's handshake page size.
    uint32_t handshake_page_size = tt::align(static_cast<uint32_t>(64), max_alignment);

    auto receiver_core_range_set = CoreRangeSet(std::set<CoreRange>());
    for (const auto& core : receiver_core_coords) {
        receiver_core_range_set = receiver_core_range_set.merge(CoreRangeSet({CoreRange(core, core)}));
    }

    uint32_t packet_header_cb_num_pages = 2;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, receiver_core_range_set, cb_packet_header_config);

    std::vector<uint32_t> handshake_compile_args = {
        packet_header_cb_index,  // fabric_packet_header_cb_id
        handshake_page_size,     // handshake_page_size (socket page size)
    };

    auto handshake_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_direct_async/device/kernels/"
        "receiver_direct.cpp",
        receiver_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(handshake_compile_args));

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& receiver_core_coord = receiver_core_coords[core_idx];
        const auto& sender_fabric_node_id = sender_fabric_node_ids[core_idx];
        const auto& receiver_fabric_node_id = receiver_fabric_node_ids[core_idx];

        std::vector<uint32_t> handshake_rt_args = {
            mesh_socket.get_config_buffer()->address(),  // socket_config_addr
            output_tensor.buffer()->address(),           // output_base_addr
            static_cast<uint32_t>(output_page_size),     // output_page_size
            static_cast<uint32_t>(total_num_pages),      // num_pages
        };

        auto link_indices = tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
        TT_FATAL(!link_indices.empty(), "No link indices found for receiver core");

        uint32_t selected_link_index = link_indices[core_idx % link_indices.size()];
        tt::tt_fabric::append_fabric_connection_rt_args(
            receiver_fabric_node_id,
            sender_fabric_node_id,
            selected_link_index,
            program,
            receiver_core_coord,
            handshake_rt_args);

        tt::tt_metal::SetRuntimeArgs(program, handshake_kernel_id, receiver_core_coord, handshake_rt_args);
    }

    return {
        std::move(program),
        shared_variables_t{
            .receiver_core_coords = receiver_core_coords,
            .handshake_kernel_id = handshake_kernel_id,
        }};
}

void RecvDirectAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RecvDirectAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        auto& receiver_core_coords = shared_vars.receiver_core_coords;
        const auto& handshake_kernel_id = shared_vars.handshake_kernel_id;

        const auto& mesh_socket = operation_attributes.mesh_socket;
        const auto& output_tensor = tensor_args;

        for (const auto& receiver_core_coord : receiver_core_coords) {
            auto& handshake_runtime_args = GetRuntimeArgs(program, handshake_kernel_id, receiver_core_coord);
            handshake_runtime_args[0] = mesh_socket.get_config_buffer()->address();
            handshake_runtime_args[1] = output_tensor.buffer()->address();
        }
    }
}

}  // namespace ttnn::experimental::prim
