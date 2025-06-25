#include "bidirectional_fabric_socket.hpp"

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

}  // namespace ttnn::distributed
