// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ax_plus_b_device_operation.hpp"

namespace ttnn::operations::examples {

AX_plus_B_DeviceOperation::program_factory_t AX_plus_B_DeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
    if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
        return SingleCore{};
    }
    return MultiCore{};
}

void AX_plus_B_DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void AX_plus_B_DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

AX_plus_B_DeviceOperation::spec_return_value_t AX_plus_B_DeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.tensor_a;
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), MemoryConfig{}));
}

AX_plus_B_DeviceOperation::tensor_return_value_t AX_plus_B_DeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.tensor_a.device());
}

std::tuple<AX_plus_B_DeviceOperation::operation_attributes_t, AX_plus_B_DeviceOperation::tensor_args_t>
AX_plus_B_DeviceOperation::invoke(
    const Tensor& tensor_a, const Tensor& tensor_x, const Tensor& tensor_b, Tensor& tensor_y) {
    return {operation_attributes_t{true, 42}, tensor_args_t{tensor_a, tensor_x, tensor_b, tensor_y}};
}

}  // namespace ttnn::operations::examples
