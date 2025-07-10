// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
    tt::tt_metal::distributed::SocketConfig socket_config) {
    if (socket_type == SocketType::MPI) {
        if (socket_config.distributed_context->rank() < other_rank) {
            socket_config.sender_rank = socket_config.distributed_context->rank();
            socket_config.receiver_rank = other_rank;
        } else {
            socket_config.sender_rank = other_rank;
            socket_config.receiver_rank = socket_config.distributed_context->rank();
        }
        auto mesh_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, socket_config);
        return std::make_unique<MPISocket>(mesh_socket);
    }

    assert(socket_type == SocketType::FABRIC);
    if (endpoint_socket_type == EndpointSocketType::SENDER) {
        socket_config.sender_rank = socket_config.distributed_context->rank();
        socket_config.receiver_rank = other_rank;
        auto mesh_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, socket_config);
        return std::make_unique<FabricSocket>(mesh_socket);
    } else if (endpoint_socket_type == EndpointSocketType::RECEIVER) {
        socket_config.sender_rank = other_rank;
        socket_config.receiver_rank = socket_config.distributed_context->rank();
        auto mesh_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, socket_config);
        return std::make_unique<FabricSocket>(mesh_socket);
    } else if (endpoint_socket_type == EndpointSocketType::BIDIRECTIONAL) {
        auto sender_socket_config = socket_config;
        sender_socket_config.sender_rank = socket_config.distributed_context->rank();
        sender_socket_config.receiver_rank = other_rank;

        auto recv_socket_config = socket_config;
        recv_socket_config.sender_rank = other_rank;
        recv_socket_config.receiver_rank = socket_config.distributed_context->rank();

        if (sender_socket_config.sender_rank == sender_socket_config.receiver_rank) {
            throw std::runtime_error("Sender and receiver ranks cannot be the same for bidirectional socket.");
        }

        // this ensures that sockets can perform handshake in correct order
        if (sender_socket_config.sender_rank < recv_socket_config.sender_rank) {
            auto send_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, sender_socket_config);
            auto recv_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, recv_socket_config);
            return std::make_unique<BidirectionalFabricSocket>(send_socket, recv_socket);
        } else {
            auto recv_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, recv_socket_config);
            auto send_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, sender_socket_config);
            return std::make_unique<BidirectionalFabricSocket>(send_socket, recv_socket);
        }

    } else {
        throw std::runtime_error("Unsupported EndpointSocketType");
    }
}

}  // namespace ttnn::distributed
