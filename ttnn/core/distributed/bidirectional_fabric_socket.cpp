// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    if (socket_config.distributed_context->rank() != socket_config.sender_rank) {
        return send_socket_.get_config().sender_rank;
    }
    return send_socket_.get_config().receiver_rank;
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
BidirectionalFabricSocket::get_distributed_context() const {
    return send_socket_.get_config().distributed_context;
}

std::unique_ptr<BidirectionalFabricSocket> BidirectionalFabricSocket::create(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank rank,
    const tt::tt_metal::distributed::SocketConfig& socket_config) {
    auto sender_socket_config = socket_config;
    sender_socket_config.sender_rank = socket_config.distributed_context->rank();
    sender_socket_config.receiver_rank = rank;

    auto recv_socket_config = socket_config;
    recv_socket_config.sender_rank = rank;
    recv_socket_config.receiver_rank = socket_config.distributed_context->rank();

    if (sender_socket_config.sender_rank == sender_socket_config.receiver_rank) {
        throw std::runtime_error("Sender and receiver ranks cannot be the same for bidirectional socket.");
    }

    // this ensures that sockets can perform handshake in correct order
    if (sender_socket_config.sender_rank < recv_socket_config.sender_rank) {
        auto send_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, sender_socket_config);
        auto recv_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, recv_socket_config);
        return std::make_unique<BidirectionalFabricSocket>(send_socket, recv_socket);
    }
    auto recv_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, recv_socket_config);
    auto send_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, sender_socket_config);
    return std::make_unique<BidirectionalFabricSocket>(send_socket, recv_socket);
}

}  // namespace ttnn::distributed
