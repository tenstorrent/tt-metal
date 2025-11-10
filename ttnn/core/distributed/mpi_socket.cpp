// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/mpi_socket.hpp"

#include <ttnn/operations/data_movement/copy/copy.hpp>

#include <stdexcept>

namespace ttnn::distributed {

namespace {

std::vector<tt::tt_metal::HostBuffer> get_as(const ttnn::Tensor& tensor) {
    return std::visit(
        [](auto&& storage) -> std::vector<tt::tt_metal::HostBuffer> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, tt::tt_metal::HostStorage>) {
                std::vector<tt::tt_metal::HostBuffer> buffers;
                buffers.reserve(storage.buffer().shard_coords().size());
                storage.buffer().apply([&buffers](const tt::tt_metal::HostBuffer& shard) { buffers.push_back(shard); });
                return buffers;
            } else {
                TT_THROW("Tensor must be on host");
            }
        },
        tensor.storage());
}

std::vector<std::span<std::byte>> get_bytes_from_cpu_tensor(ttnn::Tensor& cpu_tensor) {
    auto buffers = get_as(cpu_tensor);

    std::vector<std::span<std::byte>> res;
    res.reserve(buffers.size());
    for (auto& buffer : buffers) {
        auto view = buffer.view_bytes();
        auto span = std::as_writable_bytes(std::span{view.begin(), view.end()});
        res.push_back(span);
    }
    return res;
}

}  // namespace

using tt::tt_metal::distributed::multihost::Tag;

MPISocket::MPISocket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::distributed::SocketConfig& socket_config) :
    mesh_device_(mesh_device), socket_config_(socket_config) {}

void MPISocket::send(const ttnn::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    auto buffers = get_bytes_from_cpu_tensor(cpu_tensor);

    auto receiver_rank = get_rank();
    for (auto buffer : buffers) {
        socket_config_.distributed_context->send(buffer, receiver_rank, Tag{0});
    }
}

void MPISocket::recv(ttnn::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    auto buffers = get_bytes_from_cpu_tensor(cpu_tensor);

    auto sender_rank = get_rank();
    for (auto buffer : buffers) {
        socket_config_.distributed_context->recv(buffer, sender_rank, Tag{0});
    }

    ttnn::assign(cpu_tensor.to_device(tensor.device()), tensor);
}

tt::tt_metal::distributed::multihost::Rank MPISocket::get_rank() const {
    auto local_rank = socket_config_.distributed_context->rank();
    if (local_rank != socket_config_.sender_rank) {
        return socket_config_.sender_rank;
    }
    return socket_config_.receiver_rank;
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> MPISocket::get_distributed_context() const {
    return socket_config_.distributed_context;
}

std::unique_ptr<MPISocket> MPISocket::create(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    tt::tt_metal::distributed::multihost::Rank rank,
    tt::tt_metal::distributed::SocketConfig socket_config) {
    if (socket_config.distributed_context->rank() < rank) {
        socket_config.sender_rank = socket_config.distributed_context->rank();
        socket_config.receiver_rank = rank;
    } else {
        socket_config.sender_rank = rank;
        socket_config.receiver_rank = socket_config.distributed_context->rank();
    }
    return std::make_unique<MPISocket>(mesh_device, socket_config);
}

}  // namespace ttnn::distributed
