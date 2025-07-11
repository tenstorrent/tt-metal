// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/bidirectional_fabric_socket.hpp"
#include "ttnn/operations/experimental/ccl/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/recv_async/recv_async.hpp"

namespace ttnn::distributed {

BidirectionalFabricSocket::BidirectionalFabricSocket(
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    const tt::tt_metal::distributed::MeshSocket& recv_socket) :
    send_socket_(send_socket), recv_socket_(recv_socket) {
    TT_FATAL(
        send_socket_.get_config().sender_rank == recv_socket_.get_config().receiver_rank &&
            send_socket_.get_config().receiver_rank == recv_socket_.get_config().sender_rank,
        "Sockets are not bidirectional");
}

void BidirectionalFabricSocket::send(const ttnn::Tensor& tensor) {
    ttnn::experimental::send_async(tensor, send_socket_);
}

void BidirectionalFabricSocket::recv(ttnn::Tensor& tensor) { ttnn::experimental::recv_async(tensor, recv_socket_); }

tt::tt_metal::distributed::multihost::Rank BidirectionalFabricSocket::get_sender_rank() const {
    return recv_socket_.get_config().sender_rank;
}

tt::tt_metal::distributed::multihost::Rank BidirectionalFabricSocket::get_receiver_rank() const {
    return send_socket_.get_config().receiver_rank;
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
BidirectionalFabricSocket::get_distributed_context() const {
    return send_socket_.get_config().distributed_context;
}

}  // namespace ttnn::distributed
