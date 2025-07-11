// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/mesh_socket.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void RecvAsync::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "recv_async op requires exactly one input tensor");
    const auto& input_tensor = input_tensors[0];
    TT_FATAL(input_tensor.device() != nullptr, "recv_async op requires a device");
    TT_FATAL(
        this->mesh_socket.get_socket_endpoint_type() == tt::tt_metal::distributed::SocketEndpoint::RECEIVER,
        "recv_async op requires a receiver socket");
    const auto* socket_mesh_device = this->mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = this->mesh_socket.get_config().socket_connection_config;
    TT_FATAL(
        this->mesh_socket.get_config().socket_mem_config.fifo_size >= input_tensor.buffer()->aligned_page_size(),
        "recv_async op requires a fifo size greater than or equal to the input tensor page size");

    auto device_ids = input_tensor.mesh_device()->get_device_ids();
    for (const auto& connection : socket_connection_config) {
        auto found_device = std::find(
            device_ids.begin(),
            device_ids.end(),
            socket_mesh_device->get_device(connection.sender_core.device_coord)->id());
        if (found_device != device_ids.end()) {
            device_ids.erase(found_device);
            if (device_ids.empty()) {
                break;
            }
        }
    }
    TT_FATAL(
        device_ids.empty(),
        "recv_async op input tensor devices {} is not part of the connected cores of the socket",
        device_ids);
}

std::vector<ttnn::TensorSpec> RecvAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors[0].tensor_spec()};
}

std::vector<Tensor> RecvAsync::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors[0]};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks RecvAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks RecvAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].mesh_device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    return recv_async_multicore(input_tensors[0], target_device, this->mesh_socket);
}

tt::tt_metal::operation::Hash RecvAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    return tt::tt_metal::operation::hash_operation<RecvAsync>(this->mesh_socket, input_tensors);
}

namespace operations::experimental::ccl {

std::vector<Tensor> recv_async_impl(
    const Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "recv_async op is only supported for Fast Dispatch");

    return tt::tt_metal::operation::run(ttnn::RecvAsync(mesh_socket), {output_tensor});
}

std::vector<Tensor> recv_async(const Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return recv_async_impl(output_tensor, mesh_socket);
}

}  // namespace operations::experimental::ccl

}  // namespace ttnn
