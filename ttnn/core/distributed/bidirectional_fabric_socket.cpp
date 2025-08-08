// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/bidirectional_fabric_socket.hpp"

namespace ttnn::distributed {

BidirectionalFabricSocket::BidirectionalFabricSocket(
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    const tt::tt_metal::distributed::MeshSocket& recv_socket) :
    send_socket_(send_socket), recv_socket_(recv_socket) {}

void BidirectionalFabricSocket::send(const ttnn::Tensor& tensor) {
    // use send_socket_ to send the tensor
    throw std::runtime_error("BidirectionalFabricSocket::send is not implemented yet. Please use MPISocket for now.");
}

void BidirectionalFabricSocket::recv(ttnn::Tensor& tensor) {
    // use recv_socket_ to receive the tensor
    throw std::runtime_error("BidirectionalFabricSocket::recv is not implemented yet. Please use MPISocket for now.");
}

tt::tt_metal::distributed::multihost::Rank BidirectionalFabricSocket::get_rank() const {
    auto socket_config = send_socket_.get_config();
    if (socket_config.distributed_context->rank() != socket_config.sender_rank) {
        return send_socket_.get_config().sender_rank;
    } else {
        return send_socket_.get_config().receiver_rank;
    }
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
BidirectionalFabricSocket::get_distributed_context() const {
    return send_socket_.get_config().distributed_context;
}

}  // namespace ttnn::distributed
