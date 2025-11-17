// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_copy_op.hpp"

#include <vector>

#include <tt-metalium/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"

namespace ttnn {

void SocketCopy::validate(const std::vector<Tensor>& input_tensors) const {}

std::vector<ttnn::TensorSpec> SocketCopy::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    // Op does not return any output tensors
    return {};
}

std::vector<Tensor> SocketCopy::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Op does not return any output tensors
    return {};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks SocketCopy::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& send_socket_connections = send_socket.get_config().socket_connection_config;
    const auto& recv_socket_connections = recv_socket.get_config().socket_connection_config;

    TT_FATAL(send_socket_connections.size() == 1, "SocketCopy only supports one sender and one receiver core.");
    TT_FATAL(recv_socket_connections.size() == 1, "SocketCopy only supports one sender and one receiver core.");
    TT_FATAL(
        send_socket_connections[0].sender_core.device_coord == recv_socket_connections[0].receiver_core.device_coord,
        "Sender and receiver cores must be on the same device.");

    TT_FATAL(
        send_socket_connections[0].sender_core.core_coord == recv_socket_connections[0].receiver_core.core_coord,
        "Sender and receiver cores must be on the same core.");

    const auto& sender_device_coord = send_socket_connections[0].sender_core.device_coord;
    ttnn::MeshCoordinateRangeSet program_coords_range_set =
        MeshCoordinateRangeSet(MeshCoordinateRange(sender_device_coord, sender_device_coord));

    return ccl::create_mesh_workload_from_programs(
        program_coords_range_set, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks SocketCopy::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return socket_copy_single_core(send_socket, recv_socket, num_bytes);
}

tt::tt_metal::operation::Hash SocketCopy::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    return tt::tt_metal::operation::hash_operation<SocketCopy>(this->recv_socket, this->send_socket, input_tensors);
}

namespace operations::experimental::ccl {

std::vector<Tensor> socket_copy(
    const Tensor& input_tensor,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    std::size_t num_bytes) {
    return tt::tt_metal::operation::run(ttnn::SocketCopy(recv_socket, send_socket, num_bytes), {input_tensor});
}

}  // namespace operations::experimental::ccl

}  // namespace ttnn
