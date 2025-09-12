// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/bidirectional_fabric_socket.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"

namespace ttnn::distributed {

BidirectionalFabricSocket::BidirectionalFabricSocket(
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    const tt::tt_metal::distributed::MeshSocket& recv_socket) :
    send_socket_(send_socket), recv_socket_(recv_socket) {}

void BidirectionalFabricSocket::send(const ttnn::Tensor& tensor) {
    ttnn::experimental::send_async(tensor, send_socket_);
}

void BidirectionalFabricSocket::recv(ttnn::Tensor& tensor) { ttnn::experimental::recv_async(tensor, recv_socket_); }

tt::tt_metal::distributed::multihost::Rank BidirectionalFabricSocket::get_rank() const {
    auto socket_config = send_socket_.get_config();
    if (*(socket_config.distributed_context->rank()) != *socket_config.sender_mesh_id) {
        return tt::tt_metal::distributed::multihost::Rank{*send_socket_.get_config().sender_mesh_id};
    } else {
        return tt::tt_metal::distributed::multihost::Rank{*send_socket_.get_config().receiver_mesh_id};
    }
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
BidirectionalFabricSocket::get_distributed_context() const {
    return send_socket_.get_config().distributed_context;
}

std::unique_ptr<BidirectionalFabricSocket> BidirectionalFabricSocket::create(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank rank,
    tt::tt_metal::distributed::SocketConfig socket_config) {
    auto sender_socket_config = socket_config;
    sender_socket_config.sender_mesh_id = tt::tt_fabric::MeshId{*(socket_config.distributed_context->rank())};
    sender_socket_config.receiver_mesh_id = tt::tt_fabric::MeshId{*rank};

    auto recv_socket_config = socket_config;
    recv_socket_config.sender_mesh_id = tt::tt_fabric::MeshId{*rank};
    recv_socket_config.receiver_mesh_id = tt::tt_fabric::MeshId{*(socket_config.distributed_context->rank())};

    if (sender_socket_config.sender_mesh_id == sender_socket_config.receiver_mesh_id) {
        throw std::runtime_error("Sender and receiver ranks cannot be the same for bidirectional socket.");
    }

    // this ensures that sockets can perform handshake in correct order
    if (sender_socket_config.sender_mesh_id < recv_socket_config.sender_mesh_id) {
        auto send_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, sender_socket_config);
        auto recv_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, recv_socket_config);
        return std::make_unique<BidirectionalFabricSocket>(send_socket, recv_socket);
    }
    auto recv_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, recv_socket_config);
    auto send_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, sender_socket_config);
    return std::make_unique<BidirectionalFabricSocket>(send_socket, recv_socket);
}

}  // namespace ttnn::distributed
