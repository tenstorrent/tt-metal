// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return_device_operation.hpp"

namespace ttnn::operations::examples {

ExampleMultipleReturnDeviceOperation::program_factory_t ExampleMultipleReturnDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SingleCore{};
}

void ExampleMultipleReturnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(attributes, tensor_args);
}

void ExampleMultipleReturnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.return_output1 || attributes.return_output2,
        "At least one output must be returned. return_output1 = {}, return_output2 = {} ",
        attributes.return_output1,
        attributes.return_output2);
}

ExampleMultipleReturnDeviceOperation::spec_return_value_t ExampleMultipleReturnDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    TensorSpec spec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), MemoryConfig{}));
    spec_return_value_t result = {std::nullopt, std::nullopt};
    if (operation_attributes.return_output1) {
        std::get<0>(result) = spec;
    }
    if (operation_attributes.return_output2) {
        std::get<1>(result) = spec;
    }
    return result;
}

ExampleMultipleReturnDeviceOperation::tensor_return_value_t ExampleMultipleReturnDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto [output1_spec_opt, output2_spec_opt] = compute_output_specs(operation_attributes, tensor_args);

    std::vector<std::optional<Tensor>> ret(2);
    if (output1_spec_opt) {
        ret[0] = create_device_tensor(*output1_spec_opt, tensor_args.input_tensor.device());
    }
    if (output2_spec_opt) {
        ret[1] = create_device_tensor(*output2_spec_opt, tensor_args.input_tensor.device());
    }

    return ret;
}

std::tuple<
    ExampleMultipleReturnDeviceOperation::operation_attributes_t,
    ExampleMultipleReturnDeviceOperation::tensor_args_t>
ExampleMultipleReturnDeviceOperation::invoke(const Tensor& input_tensor, bool return_output1, bool return_output2) {
    return {operation_attributes_t{true, 42, return_output1, return_output2}, tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::examples
