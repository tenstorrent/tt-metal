// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

RecvAsyncMeshWorkloadFactory::cached_mesh_workload_t RecvAsyncMeshWorkloadFactory::create_mesh_workload(
    const RecvAsyncParams& operation_attributes,
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

ttnn::device_operation::CachedProgram<RecvAsyncMeshWorkloadFactory::shared_variables_t>
RecvAsyncMeshWorkloadFactory::create_at(
    const RecvAsyncParams& operation_attributes,
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

    CoreCoord receiver_core_coord;
    tt::tt_fabric::FabricNodeId sender_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::FabricNodeId receiver_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};

    // TODO #24995: Find appropriate receiver core and fabric node IDs based on mesh socket configuration
    for (const auto& connection : socket_connection_config) {
        if (socket_mesh_device->get_device(connection.receiver_core.device_coord)->id() == target_device->id()) {
            receiver_core_coord = connection.receiver_core.core_coord;
            receiver_fabric_node_id = output_tensor.device()->get_fabric_node_id(connection.receiver_core.device_coord);
            sender_fabric_node_id = mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::SENDER, connection.sender_core.device_coord);
            break;
        }
    }
    // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        output_tensor.buffer()->alignment());
    auto output_page_size = output_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(output_page_size, max_alignment);
    auto fabric_max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    uint32_t socket_block_size = socket_aligned_page_size;
    uint32_t packet_header_cb_num_pages = 1;
    uint32_t packet_header_cb_page_size = fabric_max_payload_size;

    auto packet_header_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, receiver_core_coord, cb_packet_header_config);

    const auto output_accessor_args = tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer());
    auto output_accessor_compile_time_args = output_accessor_args.get_compile_time_args();

    auto link_indices = tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);
    TT_FATAL(link_indices.size(), "Expected at least one routing plane between sender and receiver.");

    uint32_t socket_fifo_size = mesh_socket.get_config().socket_mem_config.fifo_size;
    uint32_t socket_fifo_size_in_pages = socket_fifo_size / socket_aligned_page_size;

    // Send a cumulative ack to the sender when half the FIFO is consumed.
    // This reduces the number of acks sent over fabric, since sending an ack
    // corresponds to 0% utilization.
    // In practice double buffering allows for:
    //  - No starvation or backpressure with better fabric utilization
    //  - Lower latency compared to sending an ack per page
    uint32_t num_pages_per_ack = std::max(socket_fifo_size_in_pages / 2, 1u);

    std::vector<uint32_t> writer_compile_args = {
        packet_header_cb_index,  // fabric_packet_header_cb_id
        output_page_size,        // output_page_size
        socket_block_size,       // socket_block_size
        num_pages_per_ack,       // num_pages_per_ack
    };
    writer_compile_args.insert(
        writer_compile_args.end(), output_accessor_compile_time_args.begin(), output_accessor_compile_time_args.end());

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/device/kernels/"
        "receiver_inplace_writer.cpp",
        receiver_core_coord,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    std::cout << "Mesh Socket config addr: " << mesh_socket.get_config_buffer()->address() << std::endl;

    std::vector<uint32_t> writer_rt_args = {
        mesh_socket.get_config_buffer()->address(), output_tensor.buffer()->address()};
    tt::tt_fabric::append_fabric_connection_rt_args(
        receiver_fabric_node_id, sender_fabric_node_id, link_indices[0], program, receiver_core_coord, writer_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel, receiver_core_coord, writer_rt_args);

    return {
        std::move(program),
        shared_variables_t{
            .receiver_core_coord = receiver_core_coord,
            .writer_kernel_id = writer_kernel,
        }};
}

void RecvAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RecvAsyncParams& operation_attributes,
    const Tensor& tensor_args,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value) {
    // Update runtime arguments for each program in the mesh workload
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        auto& receiver_core_coord = shared_vars.receiver_core_coord;
        const auto& writer_kernel_id = shared_vars.writer_kernel_id;
        const auto& mesh_socket = operation_attributes.mesh_socket;
        const auto& output_tensor = tensor_args;

        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, receiver_core_coord);

        writer_runtime_args[0] = mesh_socket.get_config_buffer()->address();
        writer_runtime_args[1] = output_tensor.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
