// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fused_persistent_moe_decode_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::fused_persistent_moe_decode {

ExecuteFusedPersistentMoeDecodeDeviceOperation::program_factory_t 
ExecuteFusedPersistentMoeDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SingleCore{};
}

void ExecuteFusedPersistentMoeDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input_tensor.layout() == Layout::TILE, "Error");
}

void ExecuteFusedPersistentMoeDecodeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attr, const tensor_args_t& args) {
    validate_on_program_cache_miss(attr, args);
}

ExecuteFusedPersistentMoeDecodeDeviceOperation::spec_return_value_t 
ExecuteFusedPersistentMoeDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& attr, const tensor_args_t& tensor_args) {
    return ttnn::TensorSpec(
        tensor_args.input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input_tensor.dtype(), 
            tt::tt_metal::PageConfig(tensor_args.input_tensor.layout()), 
            attr.output_mem_config));
}

ExecuteFusedPersistentMoeDecodeDeviceOperation::tensor_return_value_t 
ExecuteFusedPersistentMoeDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& attr, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attr, args), args.input_tensor.device());
}

std::tuple<ExecuteFusedPersistentMoeDecodeDeviceOperation::operation_attributes_t, ExecuteFusedPersistentMoeDecodeDeviceOperation::tensor_args_t>
ExecuteFusedPersistentMoeDecodeDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& topk_expert_indices,
    const Tensor& topk_expert_weights,
    const Tensor& w1_experts,
    const Tensor& w3_experts,
    const Tensor& w2_experts) {
    return {
        operation_attributes_t{input_tensor.memory_config()},
        tensor_args_t{input_tensor, topk_expert_indices, topk_expert_weights, w1_experts, w3_experts, w2_experts}
    };
}

} // namespace ttnn::operations::experimental::fused_persistent_moe_decode
