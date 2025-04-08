// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_device_operation.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::experimental::reduction {

// the result depends on tensor_args!
CumprodDeviceOperation::program_factory_t CumprodDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return CumprodProgramFactory{};
}

void CumprodDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void CumprodDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

CumprodDeviceOperation::spec_return_value_t CumprodDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_out.has_value()) {
        return tensor_args.optional_out->get_tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (attributes.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input_tensor.get_layout();
    }

    const auto output_shape = tensor_args.input_tensor.logical_shape();
    return TensorSpec(
        output_shape,
        TensorLayout(tensor_args.input_tensor.get_dtype(), output_layout, attributes.output_memory_config));
}

CumprodDeviceOperation::tensor_return_value_t CumprodDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_out.has_value()) {
        return *tensor_args.optional_out;  // a new Python reference to the same tensor is returned here
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

CumprodDeviceOperation::invocation_result_t CumprodDeviceOperation::invoke(
    const Tensor& input_tensor,
    const int32_t dim,
    std::optional<Tensor> optional_out,
    const MemoryConfig& memory_config,
    const QueueId& queue_id) {
    return {operation_attributes_t{dim, memory_config}, tensor_args_t{input_tensor, std::move(optional_out)}};
}

}  // namespace ttnn::operations::experimental::reduction
