// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>
#include <enchantum/enchantum.hpp>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::send_recv_utils {
template <tt::tt_metal::distributed::SocketEndpoint socket_type>
void validate(
    const std::vector<ttnn::Tensor>& input_tensors,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const std::string& op_name) {
    TT_FATAL(input_tensors.size() == 1, "{} op requires exactly one input tensor", op_name);
    const auto& input_tensor = input_tensors[0];
    TT_FATAL(input_tensor.device() != nullptr, "{} op requires a device", op_name);
    TT_FATAL(
        mesh_socket.get_socket_endpoint_type() == socket_type,
        "{} op requires a {} socket",
        op_name,
        enchantum::to_string(socket_type));
    TT_FATAL(
        mesh_socket.get_config().socket_mem_config.fifo_size >= input_tensor.buffer()->aligned_page_size(),
        "{} op requires a fifo size greater than or equal to the input tensor page size",
        op_name);
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
