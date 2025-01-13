// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sampling_op.hpp"
#include "sampling_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction {

void Sampling::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_values_tensor = input_tensors.at(0);
    const auto& input_indices_tensor = input_tensors.at(1);

    TT_FATAL(
        input_values_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(input_values_tensor.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_values_tensor.get_layout() == Layout::TILE, "Only TILE_LAYOUT is supported for inputs!");

    TT_FATAL(
        input_indices_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(input_indices_tensor.get_dtype() == DataType::UINT32, "Only UINT32 is supported for outputs!");
    TT_FATAL(input_indices_tensor.get_layout() == Layout::TILE, "Only TILE_LAYOUT is supported for inputs!");

    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");
    const auto& optional_output_tensor = output_tensors.at(0);
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            optional_output_tensor.value().get_dtype() == DataType::UINT32, "Only UINT32 is supported for outputs!");
        TT_FATAL(
            optional_output_tensor.value().memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");
    }
}

std::vector<TensorSpec> Sampling::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0)->get_tensor_spec()};
    }

    const auto& input_values_tensor = input_tensors[0];
    auto input_shape = input_values_tensor.get_logical_shape();
    ttnn::SimpleShape output_shape({1, 1, input_shape[2], 1});

    return {TensorSpec(
        output_shape,
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), input_values_tensor.memory_config()))};
}

std::vector<Tensor> Sampling::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors[0].device())};
}

operation::ProgramWithCallbacks Sampling::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_values_tensor = input_tensors.at(0);
    const auto& input_indices_tensor = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);
    return detail::sampling_multicore_interleaved(input_values_tensor, input_indices_tensor, k, p, output_tensor);
}

}  // namespace ttnn::operations::reduction
