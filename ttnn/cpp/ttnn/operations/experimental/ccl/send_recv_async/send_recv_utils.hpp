// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace ttnn::send_recv_utils {

uint32_t handshake_page_size(uint32_t max_alignment);

tt::tt_metal::IDevice* resolve_target_device(
    const Tensor& tensor, const std::optional<ttnn::MeshCoordinate>& coord, const std::string& op_name);

uint32_t socket_max_alignment(const ttnn::Tensor& tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

void validate_fifo_size(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket, const std::string& op_name, uint32_t min_fifo_size);

template <tt::tt_metal::distributed::SocketEndpoint socket_type>
void validate(
    const std::vector<ttnn::Tensor>& input_tensors,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const std::string& op_name,
    std::optional<uint32_t> min_fifo_size = std::nullopt);

template <tt::tt_metal::distributed::SocketEndpoint socket_type>
ttnn::MeshCoordinateRangeSet get_workload_coords(
    const ttnn::MeshCoordinateRangeSet& tensor_coords, const tt::tt_metal::distributed::MeshSocket& mesh_socket);

}  // namespace ttnn::send_recv_utils
