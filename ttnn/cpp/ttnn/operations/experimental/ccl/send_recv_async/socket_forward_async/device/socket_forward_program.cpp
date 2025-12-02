// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_forward_op.hpp"

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

using namespace tt::constants;

namespace ttnn {

tt::tt_metal::operation::ProgramWithCallbacks socket_forward_minimal_single_core_multi_link(
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    std::size_t num_bytes) {
    tt::tt_metal::Program program{};
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
    uint32_t num_fwd_links = 2;
    uint32_t num_bwd_links = 1;

    auto fabric_max_payload_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    uint32_t partial_packet_size = num_bytes % fabric_max_payload_size;
    uint32_t socket_block_size = socket_aligned_page_size;

    uint32_t num_whole_packets = num_bytes / fabric_max_payload_size;
    uint32_t num_whole_packets_link_0 =
        (num_whole_packets / num_fwd_links) + static_cast<uint32_t>(partial_packet_size > 0);
    uint32_t num_whole_packets_link_1 = num_whole_packets - num_whole_packets_link_0;

    uint32_t packet_header_cb_num_pages = num_fwd_links + num_bwd_links;
    uint32_t packet_header_cb_page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto packet_header_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig cb_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_num_pages * packet_header_cb_page_size, {{packet_header_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_cb_page_size);

    CreateCircularBuffer(program, my_core_coord, cb_packet_header_config);

    std::vector<uint32_t> compile_args = {
        packet_header_cb_index,
        socket_block_size,
        partial_packet_size,
        fabric_max_payload_size,
        num_whole_packets_link_0,
        num_whole_packets_link_1,
    };

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/socket_forward_async/device/kernels/"
        "socket_forward.cpp",
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

    auto override_runtime_arguments_callback =
        [](const void* operation,
           tt::tt_metal::Program& program,
           const std::vector<Tensor>& input_tensors,
           const std::vector<std::optional<const Tensor>>& optional_input_tensors,
           const std::vector<Tensor>& output_tensors) {};

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
