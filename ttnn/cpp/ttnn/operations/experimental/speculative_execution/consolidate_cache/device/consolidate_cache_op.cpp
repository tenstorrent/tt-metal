// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "consolidate_cache_op.hpp"

#include "consolidate_cache_program_factory.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::speculative_execution {

void ConsolidateCache::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 4, "ConsolidateCache requires 4 input tensors");
    auto& input = input_tensors.at(0);
    auto& other = input_tensors.at(1);
    auto& priority = input_tensors.at(2);
    auto& other_priority = input_tensors.at(3);
    TT_FATAL(
        input.get_dtype() == other.get_dtype(),
        "Input and other tensors must have the same dtype, got {} and {}",
        input.get_dtype(),
        other.get_dtype());
    TT_FATAL(
        input.get_logical_shape() == other.get_logical_shape(),
        "Input and other tensors must have the same shape, got {} and {}",
        input.get_logical_shape(),
        other.get_logical_shape());
    TT_FATAL(
        input.layout() == other.layout(),
        "Input and other tensors must have the same layout, got {} and {}",
        input.layout(),
        other.layout());
    TT_FATAL(
        input.memory_config() == other.memory_config(),
        "Input and other tensors must have the same memory config, got {} and {}",
        input.memory_config(),
        other.memory_config());
    TT_FATAL(
        priority.get_logical_shape() == other_priority.get_logical_shape(),
        "Priority and other_priority tensors must have the same shape, got {} and {}",
        priority.get_logical_shape(),
        other_priority.get_logical_shape());
    TT_FATAL(
        priority.get_dtype() == other_priority.get_dtype(),
        "Priority and other_priority tensors must have the same dtype, got {} and {}",
        priority.get_dtype(),
        other_priority.get_dtype());
    TT_FATAL(
        priority.get_dtype() == DataType::UINT32,
        "Priority tensor must have dtype float32, got {}",
        priority.get_dtype());
}

std::vector<TensorSpec> ConsolidateCache::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    auto output =
        TensorSpec(input.get_logical_shape(), TensorLayout(input.get_dtype(), input.layout(), input.memory_config()));
    return {output};
}

operation::ProgramWithCallbacks ConsolidateCache::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& input_tensor = input_tensors.at(0);
    auto& other_tensor = input_tensors.at(1);
    auto& priority_tensor = input_tensors.at(2);
    auto& other_priority_tensor = input_tensors.at(3);
    auto& output_tensor = output_tensors.at(0);
    return detail::consolidate_cache(input_tensor, other_tensor, priority_tensor, other_priority_tensor, output_tensor);
}

}  // namespace ttnn::operations::experimental::speculative_execution
