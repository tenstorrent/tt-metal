// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/create_socket.hpp"
#include "ttnn/distributed/bidirectional_fabric_socket.hpp"
#include "ttnn/distributed/fabric_socket.hpp"
#include "ttnn/distributed/mpi_socket.hpp"
#include "tt-metalium/distributed_context.hpp"

namespace ttnn::distributed {

std::unique_ptr<ISocket> create_socket(
    SocketType socket_type,
    EndpointSocketType endpoint_socket_type,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank other_rank,
    const tt::tt_metal::distributed::SocketConfig& socket_config) {
    if (socket_type == SocketType::MPI) {
        return MPISocket::create(mesh_device, other_rank, socket_config);
    }

    TT_FATAL(
        socket_type == SocketType::FABRIC, "Only FABRIC and MPI socket types are supported. Received: {}", socket_type);
    if (endpoint_socket_type == EndpointSocketType::SENDER) {
        return FabricSocket::create(mesh_device, socket_config.distributed_context->rank(), other_rank, socket_config);
    }
    if (endpoint_socket_type == EndpointSocketType::RECEIVER) {
        return FabricSocket::create(mesh_device, other_rank, socket_config.distributed_context->rank(), socket_config);
    }
    if (endpoint_socket_type == EndpointSocketType::BIDIRECTIONAL) {
        return BidirectionalFabricSocket::create(mesh_device, other_rank, socket_config);
    }
    throw std::runtime_error("Unsupported EndpointSocketType");
}

}  // namespace ttnn::distributed
