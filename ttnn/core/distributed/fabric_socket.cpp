// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/fabric_socket.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include <stdexcept>

namespace ttnn::distributed {

namespace {

bool check_if_send_socket(const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    const auto& socket_config = mesh_socket.get_config();
    auto expected_sender_rank = socket_config.distributed_context->rank();
    return (socket_config.sender_rank == expected_sender_rank);
}

bool check_if_recv_socket(const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    const auto& socket_config = mesh_socket.get_config();
    auto expected_receiver_rank = socket_config.distributed_context->rank();
    return (socket_config.receiver_rank == expected_receiver_rank);
}

}  // namespace

FabricSocket::FabricSocket(const tt::tt_metal::distributed::MeshSocket& mesh_socket) : mesh_socket_(mesh_socket) {}

void FabricSocket::send(const ttnn::Tensor& tensor) {
    assert(check_if_send_socket(mesh_socket_));
    ttnn::experimental::send_async(tensor, mesh_socket_);
}

void FabricSocket::recv(ttnn::Tensor& tensor) {
    assert(check_if_recv_socket(mesh_socket_));
    ttnn::experimental::recv_async(tensor, mesh_socket_);
}

tt::tt_metal::distributed::multihost::Rank FabricSocket::get_rank() const {
    if (check_if_send_socket(mesh_socket_)) {
        return mesh_socket_.get_config().sender_rank;
    } else if (check_if_recv_socket(mesh_socket_)) {
        return mesh_socket_.get_config().receiver_rank;
    }

    TT_THROW(
        "FabricSocket must be either a sender or a receiver socket. "
        "Check if the socket is configured correctly.");
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> FabricSocket::get_distributed_context()
    const {
    return mesh_socket_.get_config().distributed_context;
}

}  // namespace ttnn::distributed
