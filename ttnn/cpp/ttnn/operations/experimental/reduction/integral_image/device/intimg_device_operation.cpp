// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_device_operation.hpp"
#include <enchantum/enchantum.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

IntImgDeviceOperation::program_factory_t IntImgDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return IntImgProgramFactory{};
}

void IntImgDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void IntImgDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

IntImgDeviceOperation::spec_return_value_t IntImgDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto output_layout{Layout::TILE};
    const auto& input_tensor = tensor_args.input_tensor;
    const auto output_shape{input_tensor.logical_shape()};
    return TensorSpec{output_shape, TensorLayout{input_tensor.dtype(), output_layout, input_tensor.memory_config()}};
}

IntImgDeviceOperation::tensor_return_value_t IntImgDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

IntImgDeviceOperation::invocation_result_t IntImgDeviceOperation::invoke(const Tensor& input_tensor) {
    return {operation_attributes_t{}, tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::experimental::reduction
