// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_device_operation.hpp"
#include "sort_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::reduction {

void SortDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensor, const std::vector<std::optional<Tensor>>& output_tensors) const {
    // Validate shapes of input and output tensors
    const auto input_tensor_shape = input_tensor.at(0).get_padded_shape();
    if (output_tensors.size() == 2) {
        if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
            const auto output_tensor_shape = output_tensors.at(0)->get_padded_shape();
            TT_FATAL(
                output_tensor_shape == input_tensor_shape,
                "Output tensor shape must be the same as input tensor shape. Got output tensor shape: {} and input "
                "tensor shape: {}",
                output_tensor_shape,
                input_tensor_shape);
            const auto output_indices_shape = output_tensors.at(1)->get_padded_shape();
            TT_FATAL(
                output_indices_shape == input_tensor_shape,
                "Output tensor indices shape must be the same as input tensor shape. Got output indices tensor shape: "
                "{} and "
                "input tensor shape: {}",
                output_indices_shape,
                input_tensor_shape);
        }
    }
}

std::vector<TensorSpec> SortDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensor, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 2) {
        if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
            return {output_tensors[0]->get_tensor_spec(), output_tensors[1]->get_tensor_spec()};
        }
    }
    // Create output tensors specs
    auto output_shape = input_tensor.at(0).get_logical_shape();
    auto values_spec = TensorSpec(
        output_shape, TensorLayout(input_tensor.at(0).get_dtype(), PageConfig(Layout::TILE), output_mem_config));
    auto index_spec =
        TensorSpec(output_shape, TensorLayout(DataType::UINT16, PageConfig(Layout::TILE), output_mem_config));

    return {values_spec, index_spec};
}

std::vector<Tensor> SortDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 2) {
        if (output_tensors.at(0).has_value() && output_tensors.at(1).has_value()) {
            return {output_tensors[0].value(), output_tensors[1].value()};
        }
    }
    auto output_specs = compute_output_specs(input_tensors, output_tensors);
    return {
        create_device_tensor(output_specs[0], input_tensors.at(0).device()),
        create_device_tensor(output_specs[1], input_tensors.at(0).device()),
    };
}

tt::tt_metal::operation::ProgramWithCallbacks SortDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return detail::sort_program_interleaved(
        input_tensors.at(0), this->dim, this->descending, this->stable, output_tensors.at(0), output_tensors.at(1));
}
}  // namespace ttnn::operations::experimental::reduction
