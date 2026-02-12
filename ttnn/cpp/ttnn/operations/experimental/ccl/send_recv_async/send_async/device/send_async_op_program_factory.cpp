// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_op_program_factory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"
using namespace tt::constants;

namespace ttnn::experimental::prim {

SendAsyncMeshWorkloadFactory::cached_mesh_workload_t SendAsyncMeshWorkloadFactory::create_mesh_workload(
    const SendAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Tensor& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    ttnn::MeshCoordinateRangeSet workload_coords =
    ttnn::send_recv_utils::get_workload_coords<tt::tt_metal::distributed::SocketEndpoint::SENDER>(
        tensor_coords, operation_attributes.mesh_socket);

    for (const auto& coord : workload_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<SendAsyncMeshWorkloadFactory::shared_variables_t>
SendAsyncMeshWorkloadFactory::create_at(
    const SendAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const Tensor& tensor_args,
    std::vector<Tensor>& /*tensor_return_value*/) {
    auto mesh_socket = operation_attributes.mesh_socket;
    auto recv_socket = operation_attributes.recv_socket.value();
    const auto& input_tensor = tensor_args;
    auto* mesh_device = input_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.device();

    tt::tt_metal::Program program{};
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;

    CoreCoord sender_core_coord;
    CoreCoord receiver_core_coord;
    tt::tt_fabric::FabricNodeId sender_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::FabricNodeId receiver_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    auto upstream_device_coord = recv_socket.get_config().socket_connection_config[0].sender_core.device_coord;
    tt::tt_fabric::FabricNodeId upstream_fabric_node_id =
        recv_socket.get_fabric_node_id(tt::tt_metal::distributed::SocketEndpoint::SENDER, upstream_device_coord);
    for (const auto& connection : socket_connection_config) {
        if (socket_mesh_device->get_device(connection.sender_core.device_coord)->id() == target_device->id()) {
            sender_core_coord = connection.sender_core.core_coord;
            receiver_core_coord = connection.receiver_core.core_coord;
            sender_fabric_node_id = input_tensor.device()->get_fabric_node_id(connection.sender_core.device_coord);
            receiver_fabric_node_id = mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::RECEIVER, connection.receiver_core.device_coord);
            break;
        }
    }

    auto max_alignment = std::max(
        target_device->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        input_tensor.buffer()->alignment());
    auto input_page_size = input_tensor.buffer()->aligned_page_size();
    auto socket_aligned_page_size = tt::align(input_page_size, max_alignment);

    auto link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, receiver_fabric_node_id);
    TT_FATAL(link_indices.size() > 1, "Single core multi link version of SendAsync only supports multiple links");
    uint32_t num_links = 1;

    uint32_t fabric_max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    uint32_t num_whole_packets = input_page_size / fabric_max_payload_size;
    uint32_t partial_packet_size = input_page_size % fabric_max_payload_size;
    uint32_t num_whole_packets_link_0 =
        (num_whole_packets / num_links) + static_cast<uint32_t>(partial_packet_size > 0);

    uint32_t socket_block_size = socket_aligned_page_size;
    uint32_t socket_fifo_size_in_pages =
        mesh_socket.get_config().socket_mem_config.fifo_size / socket_aligned_page_size;

    uint32_t cb_num_pages = socket_fifo_size_in_pages;
    uint32_t cb_page_size = socket_block_size;

    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    auto src0_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, cb_page_size);

    CreateCircularBuffer(program, sender_core_coord, cb_src0_config);

    uint32_t packet_header_cb_num_pages = num_links + 1;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, sender_core_coord, cb_packet_header_config);

    const auto input_accessor_args = tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer());
    auto compile_time_args = input_accessor_args.get_compile_time_args();

    constexpr uint32_t barrier_address = 1105600;
    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,             // cb0_id
        packet_header_cb_index,    // fabric_packet_header_cb_id
        socket_block_size,         // socket_block_size
        partial_packet_size,       // partial_packet_size
        fabric_max_payload_size,   // whole_packet_size (fabric_max_payload_size)
        num_whole_packets_link_0,  // num_whole_packets_link_0
        input_page_size,           // input_page_size
        barrier_address,
    };
    writer_compile_args.insert(writer_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    auto worker_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/kernels/sender_writer.cpp",
        sender_core_coord,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // TODO #24995: These parameters should be derived from the expected tensor/socket configuration
    uint32_t bank_id = target_device->allocator()->get_bank_ids_from_logical_core(
        mesh_socket.get_config().socket_mem_config.socket_storage_type, receiver_core_coord)[0];

    std::vector<uint32_t> writer_rt_args = {
        input_tensor.buffer()->address(),
        mesh_socket.get_config_buffer()->address(),
        recv_socket.get_config_buffer()->address(),
        bank_id};

    for (uint32_t i = 0; i < num_links; i++) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_fabric_node_id,
            receiver_fabric_node_id,
            link_indices[i],
            program,
            sender_core_coord,
            writer_rt_args);
    }

    auto bwd_link_indices = tt::tt_fabric::get_forwarding_link_indices(sender_fabric_node_id, upstream_fabric_node_id);
    tt::tt_fabric::append_fabric_connection_rt_args(
        sender_fabric_node_id,
        upstream_fabric_node_id,
        bwd_link_indices[0],
        program,
        sender_core_coord,
        writer_rt_args);

    tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel_id, sender_core_coord, writer_rt_args);

    return {
        std::move(program),
        shared_variables_t{
            .sender_core_coord = sender_core_coord,
            .worker_writer_kernel_id = worker_writer_kernel_id,
        }};
}

void SendAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const SendAsyncParams& operation_attributes,
    const Tensor& /*tensor_args*/,
    [[maybe_unused]] std::vector<Tensor>& tensor_return_value) {
    // Update runtime arguments for each program in the mesh workload
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        auto& sender_core_coord = shared_vars.sender_core_coord;
        // const auto& worker_reader_kernel_id = shared_vars.worker_reader_kernel_id;
        const auto& worker_writer_kernel_id = shared_vars.worker_writer_kernel_id;

        const auto& mesh_socket = operation_attributes.mesh_socket;
        const auto& recv_socket = operation_attributes.recv_socket.value();
        // const auto& input_tensor = tensor_args;

        // auto& reader_runtime_args = GetRuntimeArgs(program, worker_reader_kernel_id, sender_core_coord);
        auto& writer_runtime_args = GetRuntimeArgs(program, worker_writer_kernel_id, sender_core_coord);

        // reader_runtime_args[0] = input_tensor.buffer()->address();
        writer_runtime_args[1] = mesh_socket.get_config_buffer()->address();
        writer_runtime_args[2] = recv_socket.get_config_buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
