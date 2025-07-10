// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/distributed/isocket.hpp"
#include "ttnn/distributed/socket_enums.hpp"
#include "tt-metalium/mesh_socket.hpp"

namespace ttnn::distributed {

std::unique_ptr<ISocket> create_socket(
    SocketType socket_type,
    EndpointSocketType endpoint_socket_type,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank other_rank,
    tt::tt_metal::distributed::SocketConfig socket_config);

}
