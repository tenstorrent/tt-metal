// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/distributed/isocket.hpp"
#include "tt-metalium/experimental/sockets/mesh_socket.hpp"

namespace ttnn::distributed {

enum class SocketType : uint8_t { MPI, FABRIC };

enum class EndpointSocketType : uint8_t { SENDER, RECEIVER, BIDIRECTIONAL };

std::unique_ptr<ISocket> create_socket(
    SocketType socket_type,
    EndpointSocketType endpoint_socket_type,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank other_rank,
    const tt::tt_metal::distributed::SocketConfig& socket_config);

}  // namespace ttnn::distributed
