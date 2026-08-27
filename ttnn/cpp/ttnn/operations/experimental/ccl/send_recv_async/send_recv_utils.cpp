// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

#include <algorithm>
#include <enchantum/enchantum.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::send_recv_utils {

// Payload of the handshake page used by the direct/buffered ops. Those ops push a single page of
// this size through the socket FIFO to exchange addresses instead of streaming tensor data through
// it, so their FIFO requirement is independent of the tensor page size.
constexpr uint32_t handshake_payload_size = 64;

uint32_t handshake_page_size(uint32_t max_alignment) { return tt::align(handshake_payload_size, max_alignment); }

tt::tt_metal::IDevice* resolve_target_device(
    const Tensor& tensor, const std::optional<ttnn::MeshCoordinate>& coord, const std::string& op_name) {
    TT_FATAL(coord.has_value(), "{}: the program factory requires a per-device mesh dispatch coordinate", op_name);
    auto* mesh_device = tensor.device();
    return mesh_device ? mesh_device->get_device(*coord) : tensor.device()->get_device(0);
}

uint32_t socket_max_alignment(const ttnn::Tensor& tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return std::max(
        tensor.buffer()->allocator()->get_alignment(mesh_socket.get_config().socket_mem_config.socket_storage_type),
        tensor.buffer()->alignment());
}

void validate_fifo_size(
    const tt::tt_metal::distributed::MeshSocket& mesh_socket, const std::string& op_name, uint32_t min_fifo_size) {
    const auto fifo_size = mesh_socket.get_config().socket_mem_config.fifo_size;
    TT_FATAL(
        fifo_size >= min_fifo_size,
        "{} op requires a fifo size of at least {} bytes, got {}",
        op_name,
        min_fifo_size,
        fifo_size);
}

// `min_fifo_size` defaults to the tensor page size, which is what the FIFO-streaming ops need. Ops
// that only push the handshake page through the FIFO pass the handshake page size instead.
template <tt::tt_metal::distributed::SocketEndpoint socket_type>
void validate(
    const std::vector<ttnn::Tensor>& input_tensors,
    const tt::tt_metal::distributed::MeshSocket& mesh_socket,
    const std::string& op_name,
    std::optional<uint32_t> min_fifo_size) {
    TT_FATAL(input_tensors.size() == 1, "{} op requires exactly one input tensor", op_name);
    const auto& input_tensor = input_tensors[0];
    TT_FATAL(input_tensor.device() != nullptr, "{} op requires a device", op_name);
    TT_FATAL(
        mesh_socket.get_socket_endpoint_type() == socket_type,
        "{} op requires a {} socket",
        op_name,
        enchantum::to_string(socket_type));
    validate_fifo_size(mesh_socket, op_name, min_fifo_size.value_or(input_tensor.buffer()->aligned_page_size()));
}

template <tt::tt_metal::distributed::SocketEndpoint socket_type>
ttnn::MeshCoordinateRangeSet get_workload_coords(
    const ttnn::MeshCoordinateRangeSet& tensor_coords, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    ttnn::MeshCoordinateRangeSet workload_coords;
    const auto& socket_connections = mesh_socket.get_config().socket_connection_config;

    const auto tensor_coords_flattened = tensor_coords.coords();
    for (const auto& connection : socket_connections) {
        const auto& device_coord = socket_type == tt::tt_metal::distributed::SocketEndpoint::SENDER
                                       ? connection.sender_core.device_coord
                                       : connection.receiver_core.device_coord;
        if (std::find(tensor_coords_flattened.begin(), tensor_coords_flattened.end(), device_coord) !=
            tensor_coords_flattened.end()) {
            workload_coords.merge(ttnn::MeshCoordinateRange(device_coord, device_coord));
        }
    }
    TT_FATAL(
        !workload_coords.empty(),
        "{} socket coordinates do not intersect with tensor coordinates.",
        (socket_type == tt::tt_metal::distributed::SocketEndpoint::SENDER ? "Sender" : "Receiver"));
    return workload_coords;
}

template void validate<tt::tt_metal::distributed::SocketEndpoint::SENDER>(
    const std::vector<ttnn::Tensor>&,
    const tt::tt_metal::distributed::MeshSocket&,
    const std::string&,
    std::optional<uint32_t>);
template void validate<tt::tt_metal::distributed::SocketEndpoint::RECEIVER>(
    const std::vector<ttnn::Tensor>&,
    const tt::tt_metal::distributed::MeshSocket&,
    const std::string&,
    std::optional<uint32_t>);
template ttnn::MeshCoordinateRangeSet get_workload_coords<tt::tt_metal::distributed::SocketEndpoint::SENDER>(
    const ttnn::MeshCoordinateRangeSet&, const tt::tt_metal::distributed::MeshSocket&);
template ttnn::MeshCoordinateRangeSet get_workload_coords<tt::tt_metal::distributed::SocketEndpoint::RECEIVER>(
    const ttnn::MeshCoordinateRangeSet&, const tt::tt_metal::distributed::MeshSocket&);

}  // namespace ttnn::send_recv_utils
