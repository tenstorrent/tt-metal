// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/barrier/device/barrier_op.hpp"

#include <cstdint>

namespace ttnn {

void Barrier::validate(const std::vector<Tensor>& input_tensors) const {
    // Validate the input tensor
    TT_FATAL(this->topology == ccl::Topology::Ring, "We currently only support Ring topologies on the barrier op");
}

std::vector<TensorSpec> Barrier::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    // For a Barrier the output shape should match the input
    std::vector<TensorSpec> result;
    result.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        result.push_back(input_tensor.tensor_spec());
    }
    return result;
}

std::vector<Tensor> Barrier::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Tensor is unmodified, return what was passed in
    return input_tensors;
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks Barrier::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks Barrier::create_program_at(
    const ttnn::MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto* target_device =
        input_tensor.mesh_device() ? input_tensor.mesh_device()->get_device(mesh_coord) : input_tensor.device();
    const auto& devices_to_use = input_tensor.mesh_device() ? input_tensor.mesh_device()->get_devices() : this->devices;

    ccl::SenderRecieverConfig config =
        ccl::get_device_sender_receiver_config(target_device, devices_to_use, this->topology);

    return ccl::barrier::detail::barrier_with_workers(
        input_tensor,
        output_tensors.at(0),
        /*is_starting_core*/ (config.device_index == 0),
        devices_to_use.size(),
        config.device_index,
        target_device->id(),
        config.receiver_device_id,
        config.sender_device_id,
        this->topology);
}

namespace operations::ccl {

Tensor barrier_function(const Tensor& input_tensor, const ttnn::Barrier& barrier_struct) {
    std::vector<Tensor> output_tensors = {input_tensor};
    return tt::tt_metal::operation::run(barrier_struct, {input_tensor}).at(0);
}

std::vector<Tensor> barrier_function(const std::vector<Tensor>& input_tensors, const ttnn::Barrier& barrier_struct) {
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        output_tensors.push_back(tt::tt_metal::operation::run(barrier_struct, {input_tensor}).at(0));
    }
    return output_tensors;
}

}  // namespace operations::ccl
}  // namespace ttnn
