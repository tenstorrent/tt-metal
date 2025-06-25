#include "fabric_socket.hpp"
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
    throw std::runtime_error("FabricSocket::send is not implemented yet. Please use MPISocket for now.");
}

void FabricSocket::recv(ttnn::Tensor& tensor) {
    assert(check_if_recv_socket(mesh_socket_));
    throw std::runtime_error("FabricSocket::recv is not implemented yet. Please use MPISocket for now.");
}

}  // namespace ttnn::distributed
