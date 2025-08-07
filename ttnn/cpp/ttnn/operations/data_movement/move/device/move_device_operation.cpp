// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void MoveDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {}

std::vector<ttnn::TensorSpec> MoveDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(1).tensor_spec()};
}

std::vector<Tensor> MoveDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return {input_tensors.at(1)};
}

operation::ProgramWithCallbacks MoveDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->move_op_parallelization_strategy;
    switch (parallelization_strategy) {
        case MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP:
            return move_multi_core_with_overlap(input_tensor, output_tensor);
        case MoveOpParallelizationStrategy::MULTI_CORE_SHARDED:
            return move_multi_core_sharded(input_tensor, output_tensor);
        case MoveOpParallelizationStrategy::MULTI_CORE:
        default: return move_multi_core(input_tensor, output_tensor);
    }
}
tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
MoveDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

MoveOpParallelizationStrategy MoveDeviceOperation::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return this->move_op_parallelization_strategy;
}

}  // namespace ttnn::operations::data_movement
