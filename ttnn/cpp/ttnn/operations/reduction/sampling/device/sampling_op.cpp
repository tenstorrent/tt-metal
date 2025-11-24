// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    const auto& k = input_tensors.at(2);
    const auto& p = input_tensors.at(3);
    const auto& temp = input_tensors.at(4);

    TT_FATAL(
        input_values_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(input_values_tensor.dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_values_tensor.layout() == Layout::TILE, "Only TILE_LAYOUT is supported for inputs!");

    TT_FATAL(
        input_indices_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(
        input_indices_tensor.dtype() == DataType::UINT32 || input_indices_tensor.dtype() == DataType::INT32,
        "Only UINT32 & INT32 dtypes are supported for input indices!");

    TT_FATAL(input_indices_tensor.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR is supported for input indices!");

    TT_FATAL(
        input_indices_tensor.logical_shape() == input_values_tensor.logical_shape(),
        "Input values and indices must have the same shape!");
    auto input_shape = input_values_tensor.logical_shape();
    TT_FATAL(input_shape[0] * input_shape[1] * input_shape[2] == 32, "Input must have 32 users!");
    TT_FATAL(input_shape[3] % 32 == 0, "Input inner dim ({}) must be divisible by 32, pad if needed!", input_shape[3]);

    if (sub_core_grids.has_value()) {
        TT_FATAL(
            sub_core_grids.value().num_cores() == input_shape[0] * input_shape[1] * input_shape[2],
            "Subcore grid expects num_users cores, but found {}!",
            sub_core_grids.value().num_cores());
    }
    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");
    const auto& optional_output_tensor = output_tensors.at(0);
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            optional_output_tensor.value().dtype() == DataType::UINT32 ||
                optional_output_tensor.value().dtype() == DataType::INT32,
            "Only UINT32 & INT32 dtypes are supported for outputs!");

        TT_FATAL(
            optional_output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");
    }

    // Check size, layout and dtype of k, p, temp
    TT_FATAL(k.dtype() == DataType::UINT32, "Only UINT32 dtypes are supported for k!");
    TT_FATAL(p.dtype() == DataType::BFLOAT16, "Only BFLOAT16 dtypes are supported for p!");
    TT_FATAL(temp.dtype() == DataType::BFLOAT16, "Only BFLOAT16 dtypes are supported for temp!");
    TT_FATAL(k.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for k!");
    TT_FATAL(p.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for p!");
    TT_FATAL(temp.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for temp!");
    TT_FATAL(k.logical_shape() == Shape({32}), "k must have shape [32]!");
    TT_FATAL(p.logical_shape() == Shape({32}), "p must have shape [32]!");
    TT_FATAL(temp.logical_shape() == Shape({32}), "temp must have shape [32]!");
}

std::vector<TensorSpec> Sampling::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0)->tensor_spec()};
    }

    const auto& input_values_tensor = input_tensors[0];
    auto input_shape = input_values_tensor.logical_shape();
    ttnn::Shape output_shape({1, 1, 1, input_shape[2]});

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
    auto& output_tensor = output_tensors.at(0);
    return detail::sampling_multicore_interleaved(input_tensors, seed, sub_core_grids, output_tensor);
}

}  // namespace ttnn::operations::reduction
