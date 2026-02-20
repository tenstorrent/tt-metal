// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/fabric_socket.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include <stdexcept>

namespace ttnn::distributed {

namespace {

namespace CMAKE_UNIQUE_NAMESPACE {

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

}  // namespace CMAKE_UNIQUE_NAMESPACE

}  // namespace

FabricSocket::FabricSocket(const tt::tt_metal::distributed::MeshSocket& mesh_socket) : mesh_socket_(mesh_socket) {}

void FabricSocket::send(const ttnn::Tensor& tensor) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    assert(check_if_send_socket(mesh_socket_));
    ttnn::experimental::send_async(tensor, mesh_socket_);
}

void FabricSocket::recv(ttnn::Tensor& tensor) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    assert(check_if_recv_socket(mesh_socket_));
    ttnn::experimental::recv_async(tensor, mesh_socket_);
}

tt::tt_metal::distributed::multihost::Rank FabricSocket::get_rank() const {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (check_if_send_socket(mesh_socket_)) {
        return mesh_socket_.get_config().sender_rank;
    }
    if (check_if_recv_socket(mesh_socket_)) {
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

std::unique_ptr<FabricSocket> FabricSocket::create(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank sender_rank,
    tt::tt_metal::distributed::multihost::Rank receiver_rank,
    tt::tt_metal::distributed::SocketConfig socket_config) {
    socket_config.sender_rank = sender_rank;
    socket_config.receiver_rank = receiver_rank;
    auto mesh_socket = tt::tt_metal::distributed::MeshSocket(mesh_device, socket_config);
    return std::make_unique<FabricSocket>(mesh_socket);
}

}  // namespace ttnn::distributed
