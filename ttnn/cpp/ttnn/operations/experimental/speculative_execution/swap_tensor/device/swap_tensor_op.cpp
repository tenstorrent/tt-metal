// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swap_tensor_op.hpp"

#include "swap_tensor_program_factory.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::speculative_execution {

void SwapTensor::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "SwapTensor requires 1 input tensor");
}

std::vector<TensorSpec> SwapTensor::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    auto output =
        TensorSpec(input.get_logical_shape(), TensorLayout(input.get_dtype(), input.layout(), input.memory_config()));
    return {output};
}

operation::ProgramWithCallbacks SwapTensor::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    TT_FATAL(this->semaphore.has_value(), "Global Semaphore is required for SwapTensor");
    return detail::swap_tensor(
        input_tensor,
        output_tensor,
        this->num_links,
        this->num_devices,
        this->device_index,
        this->topology,
        this->semaphore.value(),
        this->forward_device,
        this->backward_device);
}

}  // namespace ttnn::operations::experimental::speculative_execution
