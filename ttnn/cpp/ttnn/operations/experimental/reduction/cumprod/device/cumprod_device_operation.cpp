// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_device_operation.hpp"

namespace ttnn::operations::experimental::reduction {

// the result depends on tensor_args!
CumprodDeviceOperation::program_factory_t CumprodDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SingleCore{};  // TODO(jbbieniek): enable a multi-core version once the implementation is ready
}

void CumprodDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void CumprodDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

CumprodDeviceOperation::spec_return_value_t CumprodDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.get_logical_shape(),
        TensorLayout(input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), MemoryConfig{}));
}

CumprodDeviceOperation::tensor_return_value_t CumprodDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

CumprodDeviceOperation::invocation_result_t CumprodDeviceOperation::invoke(
    const Tensor& input_tensor, const int64_t dim) {
    // TODO(jbbieniek): finish this
    return {operation_attributes_t{dim}, tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::experimental::reduction
