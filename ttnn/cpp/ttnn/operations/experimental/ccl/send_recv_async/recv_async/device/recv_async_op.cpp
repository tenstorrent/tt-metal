// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "recv_async_op.hpp"

#include <vector>

#include <tt-metalium/mesh_socket.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn {

void RecvAsync::validate(const std::vector<Tensor>& input_tensors) const {
    ttnn::send_recv_utils::validate<tt::tt_metal::distributed::SocketEndpoint::RECEIVER>(
        input_tensors, this->mesh_socket, "recv_async");
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
    return tt::tt_metal::operation::run(ttnn::RecvAsync(mesh_socket), {output_tensor});
}

std::vector<Tensor> recv_async(const Tensor& output_tensor, const tt::tt_metal::distributed::MeshSocket& mesh_socket) {
    return recv_async_impl(output_tensor, mesh_socket);
}

}  // namespace operations::experimental::ccl

}  // namespace ttnn
