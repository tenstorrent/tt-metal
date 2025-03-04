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
        result.push_back(input_tensor.get_tensor_spec());
    }
    return result;
}

std::vector<Tensor> Barrier::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Tensor is unmodified, return what was passed in
    return input_tensors;
}

tt::tt_metal::operation::ProgramWithCallbacks Barrier::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::barrier::detail::barrier_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->is_starting_core,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology);
}

void Barrier::update_structure(const Tensor& input_tensor) {
    // Need to resolve the neighbours of this tensor
    // Can only be done in launch_op as it differs for different input tensors
    const auto devices = this->devices;
    const bool is_linear = (topology == ttnn::ccl::Topology::Linear);
    const uint32_t num_devices = this->ring_size;
    uint32_t device_index = 0;
    std::optional<chip_id_t> receiver_device_id = std::nullopt;
    std::optional<chip_id_t> sender_device_id = std::nullopt;
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            bool is_last_chip_in_clockwise_direction = is_linear && i == (num_devices - 1);
            bool is_last_chip_in_counter_clockwise_direction = is_linear && i == 0;
            device_index = i;
            receiver_device_id = is_last_chip_in_clockwise_direction
                                     ? std::nullopt
                                     : std::optional<chip_id_t>(devices.at((i + 1) % num_devices)->id());
            sender_device_id = is_last_chip_in_counter_clockwise_direction
                                   ? std::nullopt
                                   : std::optional<chip_id_t>(devices.at((i + num_devices - 1) % num_devices)->id());
            break;
        }
    }
    this->receiver_device_id = receiver_device_id;
    this->sender_device_id = sender_device_id;
    this->ring_index = device_index;
    this->is_starting_core = device_index == 0;
}
namespace operations::ccl {

Tensor barrier_function(const Tensor& input_tensor, const ttnn::Barrier& barrier_struct) {
    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
    tt::tt_metal::operation::launch_op(
        [barrier_struct](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const Tensor& input_tensor = input_tensors.at(0);
            // need to copy and update barrier struct for this particular tensor
            ttnn::Barrier new_barrier_struct = barrier_struct;
            new_barrier_struct.update_structure(input_tensor);
            return tt::tt_metal::operation::run(new_barrier_struct, {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace operations::ccl
}  // namespace ttnn
