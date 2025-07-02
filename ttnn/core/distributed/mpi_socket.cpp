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
                return {storage.buffer};
            } else if constexpr (std::is_same_v<StorageType, tt::tt_metal::MultiDeviceHostStorage>) {
                std::vector<tt::tt_metal::HostBuffer> buffers;
                buffers.reserve(storage.distributed_buffer().shard_coords().size());
                storage.distributed_buffer().apply(
                    [&buffers](const tt::tt_metal::HostBuffer& shard) { buffers.push_back(shard); });
                return buffers;
            } else {
                throw std::runtime_error("Tensor must be on host");
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

MPISocket::MPISocket(const tt::tt_metal::distributed::MeshSocket& mesh_socket) : mesh_socket_(mesh_socket) {}

void MPISocket::send(const ttnn::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    auto buffers = get_bytes_from_cpu_tensor(cpu_tensor);

    const auto& socket_config = mesh_socket_.get_config();

    auto rank = socket_config.distributed_context->rank();
    auto receiver_rank = socket_config.receiver_rank;
    if (rank == receiver_rank) {
        receiver_rank = socket_config.sender_rank;
    }

    for (auto buffer : buffers) {
        socket_config.distributed_context->send(buffer, receiver_rank, Tag{0});
    }
}

void MPISocket::recv(ttnn::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    auto buffers = get_bytes_from_cpu_tensor(cpu_tensor);

    const auto& socket_config = mesh_socket_.get_config();

    auto rank = socket_config.distributed_context->rank();
    auto sender_rank = socket_config.sender_rank;
    if (rank == sender_rank) {
        sender_rank = socket_config.receiver_rank;
    }

    for (auto buffer : buffers) {
        socket_config.distributed_context->recv(buffer, sender_rank, Tag{0});
    }

    ttnn::assign(cpu_tensor.to_device(tensor.device()), tensor);
}

tt::tt_metal::distributed::multihost::Rank MPISocket::get_sender_rank() const {
    return mesh_socket_.get_config().sender_rank;
}

tt::tt_metal::distributed::multihost::Rank MPISocket::get_receiver_rank() const {
    return mesh_socket_.get_config().receiver_rank;
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> MPISocket::get_distributed_context() const {
    return mesh_socket_.get_config().distributed_context;
}

}  // namespace ttnn::distributed
