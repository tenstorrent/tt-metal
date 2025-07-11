// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "recv_async_op.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include <ttnn/tensor/tensor_accessor_args.hpp>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks recv_async_multicore(
    const Tensor& output_tensor, IDevice* target_device, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    tt::tt_metal::Program program{};
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;
    CoreCoord receiver_core_coord;
    tt::tt_fabric::FabricNodeId sender_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};
    tt::tt_fabric::FabricNodeId receiver_fabric_node_id{tt::tt_fabric::MeshId{0}, 0};

    // TODO: Find appropriate receiver core and fabric node IDs based on mesh socket configuration
    for (const auto& connection : socket_connection_config) {
        if (socket_mesh_device->get_device(connection.receiver_core.device_coord)->id() == target_device->id()) {
            receiver_core_coord = connection.receiver_core.core_coord;
            receiver_fabric_node_id =
                output_tensor.mesh_device()->get_device_fabric_node_id(connection.sender_core.device_coord);
            sender_fabric_node_id = mesh_socket.get_fabric_node_id(
                tt::tt_metal::distributed::SocketEndpoint::SENDER, connection.sender_core.device_coord);
            break;
        }
    }

    // TODO: These parameters should be derived from the expected tensor/socket configuration
    auto aligned_page_size = output_tensor.buffer()->aligned_page_size();
    auto num_pages = output_tensor.buffer()->num_pages();
    auto fabric_max_payload_size = tt::round_down(
        std::min(
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(),
            static_cast<size_t>(mesh_socket.get_config().socket_mem_config.fifo_size)),
        output_tensor.buffer()->alignment());
    auto num_pages_per_packet = fabric_max_payload_size / aligned_page_size;
    uint32_t num_whole_packets = 0, num_pages_remainder = 0, num_whole_packets_per_page = 0, partial_packet_size = 0;
    if (num_pages_per_packet > 0) {
        num_whole_packets = num_pages / num_pages_per_packet;
        num_pages_remainder = num_pages % num_pages_per_packet;
    }
    if (aligned_page_size > fabric_max_payload_size) {
        num_whole_packets_per_page = aligned_page_size / fabric_max_payload_size;
        partial_packet_size = aligned_page_size % fabric_max_payload_size;
    }

    uint32_t packet_header_cb_num_pages = 1;  // One for sync
    uint32_t packet_header_cb_page_size = fabric_max_payload_size;

    auto packet_header_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    tt::tt_metal::CBHandle cb_packet_header_worker =
        CreateCircularBuffer(program, receiver_core_coord, cb_packet_header_config);

    const auto output_accessor_args = tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer());
    auto compile_time_args = output_accessor_args.get_compile_time_args();

    std::vector<uint32_t> writer_compile_args = {
        packet_header_cb_index,      // fabric_packet_header_cb_id
        num_pages,                   // num_pages
        aligned_page_size,           // page_size
        num_pages_per_packet,        // num_pages_per_packet
        num_whole_packets,           // num_whole_packets
        num_pages_remainder,         // num_pages_remainder
        num_whole_packets_per_page,  // num_whole_packets_per_page
        partial_packet_size,         // partial_packet_size
        fabric_max_payload_size,     // fabric_max_payload_size
    };
    writer_compile_args.insert(writer_compile_args.end(), compile_time_args.begin(), compile_time_args.end());

    auto worker_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/recv_async/device/kernels/receiver_writer.cpp",
        receiver_core_coord,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    std::vector<uint32_t> writer_rt_args = {
        mesh_socket.get_config_buffer()->address(), output_tensor.buffer()->address()};

    auto link_indices = tt::tt_fabric::get_forwarding_link_indices(receiver_fabric_node_id, sender_fabric_node_id);

    tt::tt_fabric::append_fabric_connection_rt_args(
        receiver_fabric_node_id, sender_fabric_node_id, link_indices[0], program, receiver_core_coord, writer_rt_args);

    tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel_id, receiver_core_coord, writer_rt_args);

    auto override_runtime_arguments_callback =
        [receiver_core_coord, worker_writer_kernel_id](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& mesh_socket = static_cast<const ttnn::RecvAsync*>(operation)->mesh_socket;
            auto& writer_runtime_args = GetRuntimeArgs(program, worker_writer_kernel_id, receiver_core_coord);

            writer_runtime_args[0] = mesh_socket.get_config_buffer()->address();
            writer_runtime_args[1] = input_tensors[0].buffer()->address();
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
