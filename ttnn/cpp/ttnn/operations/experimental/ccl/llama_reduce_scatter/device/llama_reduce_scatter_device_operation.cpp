// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "cpp/ttnn/tensor/types.hpp"
#include "llama_reduce_scatter_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

LlamaReduceScatterDeviceOperation::program_factory_t LlamaReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return LlamaReduceScatterAdd{};  // When both the tiled dimensions are moved
}

void LlamaReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void LlamaReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

LlamaReduceScatterDeviceOperation::spec_return_value_t LlamaReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->get_tensor_spec();
    }
    return tensor_args.input_tensor.get_tensor_spec();
}

LlamaReduceScatterDeviceOperation::tensor_return_value_t LlamaReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<LlamaReduceScatterDeviceOperation::operation_attributes_t, LlamaReduceScatterDeviceOperation::tensor_args_t>
LlamaReduceScatterDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor, const int32_t dim, const std::optional<ttnn::MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{.dim = dim, .output_mem_config = memory_config.value_or(input_tensor.memory_config())},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)}};
}

}  // namespace ttnn::operations::experimental::ccl
