// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"

namespace ttnn::operations::examples {

ExampleDeviceOperation::program_factory_t ExampleDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
    if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
        return SingleCore{};
    }
    return MultiCore{};
}

void ExampleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void ExampleDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

ExampleDeviceOperation::spec_return_value_t ExampleDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.input_tensor.get_tensor_spec();
}

ExampleDeviceOperation::tensor_return_value_t ExampleDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<ExampleDeviceOperation::operation_attributes_t, ExampleDeviceOperation::tensor_args_t>
ExampleDeviceOperation::invoke(const Tensor& input_tensor) {
    return {operation_attributes_t{true, 42}, tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::examples
