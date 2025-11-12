// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/bidirectional_fabric_socket.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"

namespace ttnn::distributed {

BidirectionalFabricSocket::BidirectionalFabricSocket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::distributed::SocketConfig& send_socket_config,
    const tt::tt_metal::distributed::SocketConfig& recv_socket_config) :
    mesh_device_(mesh_device), send_socket_config_(send_socket_config), recv_socket_config_(recv_socket_config) {}

void BidirectionalFabricSocket::send(const ttnn::Tensor& tensor) {
    // ttnn::experimental::send_async(tensor, mesh_device_, send_socket_config_);
}

void BidirectionalFabricSocket::recv(ttnn::Tensor& tensor) {
    // ttnn::experimental::recv_async(tensor, mesh_device_, recv_socket_config_);
}

tt::tt_metal::distributed::multihost::Rank BidirectionalFabricSocket::get_rank() const {
    if (send_socket_config_.distributed_context->rank() != send_socket_config_.sender_rank) {
        return send_socket_config_.sender_rank;
    } else {
        return send_socket_config_.receiver_rank;
    }
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
BidirectionalFabricSocket::get_distributed_context() const {
    return send_socket_config_.distributed_context;
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
        return std::make_unique<BidirectionalFabricSocket>(mesh_device, sender_socket_config, recv_socket_config);
    }
    // else
    return std::make_unique<BidirectionalFabricSocket>(mesh_device, sender_socket_config, recv_socket_config);
}

}  // namespace ttnn::distributed
