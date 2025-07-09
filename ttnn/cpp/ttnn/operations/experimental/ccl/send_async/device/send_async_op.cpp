// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "send_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/mesh_socket.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void SendAsync::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "send_async op requires exactly one input tensor");
    const auto& input_tensor = input_tensors[0];
    TT_FATAL(input_tensor.device() != nullptr, "send_async op requires a device");
    TT_FATAL(
        this->mesh_socket.get_socket_endpoint_type() == tt::tt_metal::distributed::SocketEndpoint::SENDER,
        "send_async op requires a sender socket");
    const auto* socket_mesh_device = this->mesh_socket.get_config_buffer()->device();
    const auto& socket_connection_config = this->mesh_socket.get_config().socket_connection_config;

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
        "send_async op input tensor devices {} is not part of the connected cores of the socket",
        device_ids);
}

std::vector<ttnn::TensorSpec> SendAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    return {};
}

std::vector<Tensor> SendAsync::create_output_tensors(const std::vector<Tensor>& input_tensors) const { return {}; }

tt::tt_metal::operation::MeshWorkloadWithCallbacks SendAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks SendAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].mesh_device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    return send_async_multicore(input_tensors[0], target_device, this->mesh_socket);
}

tt::tt_metal::operation::Hash SendAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    return tt::tt_metal::operation::hash_operation<SendAsync>(this->mesh_socket, input_tensors);
}

namespace operations::experimental::ccl {

std::vector<Tensor> send_async_impl(
    const Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "send_async op is only supported for Fast Dispatch");

    return tt::tt_metal::operation::run(ttnn::SendAsync(mesh_socket), {input_tensor});
}

std::vector<Tensor> send_async(const Tensor& input_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return send_async_impl(input_tensor, mesh_socket);
}

}  // namespace operations::experimental::ccl

}  // namespace ttnn
