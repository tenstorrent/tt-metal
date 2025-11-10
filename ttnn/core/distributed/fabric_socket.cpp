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

bool check_if_send_socket(const tt::tt_metal::distributed::SocketConfig& socket_config) {
    auto expected_sender_rank = socket_config.distributed_context->rank();
    return (socket_config.sender_rank == expected_sender_rank);
}

bool check_if_recv_socket(const tt::tt_metal::distributed::SocketConfig& socket_config) {
    auto expected_receiver_rank = socket_config.distributed_context->rank();
    return (socket_config.receiver_rank == expected_receiver_rank);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

}  // namespace

FabricSocket::FabricSocket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::distributed::SocketConfig& socket_config) :
    mesh_device_(mesh_device), socket_config_(socket_config) {}

void FabricSocket::send(const ttnn::Tensor& tensor) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    assert(check_if_send_socket(socket_config_));
    ttnn::experimental::send_async(tensor, mesh_device_, socket_config_);
}

void FabricSocket::recv(ttnn::Tensor& tensor) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    assert(check_if_recv_socket(socket_config_));
    ttnn::experimental::recv_async(tensor, mesh_device_, socket_config_);
}

tt::tt_metal::distributed::multihost::Rank FabricSocket::get_rank() const {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    if (check_if_send_socket(socket_config_)) {
        return socket_config_.sender_rank;
    } else if (check_if_recv_socket(socket_config_)) {
        return socket_config_.receiver_rank;
    }

    TT_THROW(
        "FabricSocket must be either a sender or a receiver socket. "
        "Check if the socket is configured correctly.");
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> FabricSocket::get_distributed_context()
    const {
    return socket_config_.distributed_context;
}

std::unique_ptr<FabricSocket> FabricSocket::create(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank sender_rank,
    tt::tt_metal::distributed::multihost::Rank receiver_rank,
    tt::tt_metal::distributed::SocketConfig socket_config) {
    socket_config.sender_rank = sender_rank;
    socket_config.receiver_rank = receiver_rank;
    return std::make_unique<FabricSocket>(mesh_device, socket_config);
}

}  // namespace ttnn::distributed
