// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_set>
#include <enchantum/enchantum.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::send_recv_utils {

// Payload of the handshake page used by the direct/buffered ops. Those ops push a single page of
// this size through the socket FIFO to exchange addresses instead of streaming tensor data through
// it, so their FIFO requirement is independent of the tensor page size.
constexpr uint32_t handshake_payload_size = 64;

// Sender and receiver must derive the same handshake page size, so both sides go through here.
inline uint32_t handshake_page_size(uint32_t max_alignment) { return tt::align(handshake_payload_size, max_alignment); }

// Alignment strict enough to satisfy both the socket storage and the tensor buffer.
inline uint32_t socket_max_alignment(
    const ttnn::Tensor& tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return std::max(
        tensor.buffer()->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        tensor.buffer()->alignment());
}

inline void validate_fifo_size(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket, const std::string& op_name, uint32_t min_fifo_size) {
    const auto fifo_size = mesh_socket.get_config().socket_mem_config.fifo_size;
    TT_FATAL(
        fifo_size >= min_fifo_size,
        "{} op requires a fifo size of at least {} bytes, got {}",
        op_name,
        min_fifo_size,
        fifo_size);
}

// `min_fifo_size` defaults to the tensor page size, which is what the FIFO-streaming ops need. Ops
// that only push the handshake page through the FIFO pass the handshake page size instead.
template <tt::tt_metal::distributed::SocketEndpoint socket_type>
void validate(
    const std::vector<ttnn::Tensor>& input_tensors,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const std::string& op_name,
    std::optional<uint32_t> min_fifo_size = std::nullopt) {
    TT_FATAL(input_tensors.size() == 1, "{} op requires exactly one input tensor", op_name);
    const auto& input_tensor = input_tensors[0];
    TT_FATAL(input_tensor.device() != nullptr, "{} op requires a device", op_name);
    TT_FATAL(
        mesh_socket.get_socket_endpoint_type() == socket_type,
        "{} op requires a {} socket",
        op_name,
        enchantum::to_string(socket_type));
    validate_fifo_size(mesh_socket, op_name, min_fifo_size.value_or(input_tensor.buffer()->aligned_page_size()));
}

template <tt::tt_metal::distributed::SocketEndpoint socket_type>
MeshCoordinateRangeSet get_workload_coords(
    const ttnn::MeshCoordinateRangeSet& tensor_coords, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    ttnn::MeshCoordinateRangeSet workload_coords;
    const auto& socket_connections = mesh_socket.get_config().socket_connection_config;

    const auto tensor_coords_flattened = tensor_coords.coords();
    for (const auto& connection : socket_connections) {
        const auto& device_coord = socket_type == tt::tt_metal::distributed::SocketEndpoint::SENDER
                                       ? connection.sender_core.device_coord
                                       : connection.receiver_core.device_coord;
        if (std::find(tensor_coords_flattened.begin(), tensor_coords_flattened.end(), device_coord) !=
            tensor_coords_flattened.end()) {
            workload_coords.merge(MeshCoordinateRange(device_coord, device_coord));
        }
    }
    TT_FATAL(
        !workload_coords.empty(),
        "{} socket coordinates do not intersect with tensor coordinates.",
        (socket_type == tt::tt_metal::distributed::SocketEndpoint::SENDER ? "Sender" : "Receiver"));
    return workload_coords;
}

}  // namespace ttnn::send_recv_utils
