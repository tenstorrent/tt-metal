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

tt::tt_metal::distributed::multihost::Rank BidirectionalFabricSocket::get_sender_rank() const {
    return send_socket_.get_config().sender_rank;
}

tt::tt_metal::distributed::multihost::Rank BidirectionalFabricSocket::get_receiver_rank() const {
    return recv_socket_.get_config().receiver_rank;
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
BidirectionalFabricSocket::get_distributed_context() const {
    return send_socket_.get_config().distributed_context;
}

}  // namespace ttnn::distributed
