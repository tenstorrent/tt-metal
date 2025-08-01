// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>
#include <enchantum/enchantum.hpp>

#include <tt-metalium/mesh_socket.hpp>
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
    const auto* socket_mesh_device = mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = mesh_socket.get_config().socket_connection_config;
    TT_FATAL(
        mesh_socket.get_config().socket_mem_config.fifo_size >= input_tensor.buffer()->aligned_page_size(),
        "{} op requires a fifo size greater than or equal to the input tensor page size",
        op_name);

    auto device_ids = input_tensor.mesh_device()->get_device_ids();
    std::unordered_set<chip_id_t> found_device_ids;
    for (const auto& connection : socket_connection_config) {
        chip_id_t device_id;
        if constexpr (socket_type == tt::tt_metal::distributed::SocketEndpoint::SENDER) {
            device_id = socket_mesh_device->get_device(connection.sender_core.device_coord)->id();
        } else {
            device_id = socket_mesh_device->get_device(connection.receiver_core.device_coord)->id();
        }
        auto found_device = std::find(device_ids.begin(), device_ids.end(), device_id);
        if (found_device != device_ids.end()) {
            found_device_ids.insert(*found_device);
            if (found_device_ids.size() == device_ids.size()) {
                break;
            }
        }
    }
    TT_FATAL(
        found_device_ids.size() == device_ids.size(),
        "{} op input tensor devices {} is not part of the connected cores of the socket",
        op_name,
        device_ids);
}

}  // namespace ttnn::send_recv_utils
