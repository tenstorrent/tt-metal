// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_add_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::onboarding {

ShardedAddOperation::program_factory_t ShardedAddOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void ShardedAddOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input_a.layout() == Layout::TILE, "input_a must be in TILE layout");
    TT_FATAL(tensor_args.input_b.layout() == Layout::TILE, "input_b must be in TILE layout");
    TT_FATAL(
        tensor_args.input_a.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED,
        "input_a must be sharded (not INTERLEAVED)");
    TT_FATAL(
        tensor_args.input_b.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED,
        "input_b must be sharded (not INTERLEAVED)");
    TT_FATAL(
        tensor_args.input_a.logical_shape() == tensor_args.input_b.logical_shape(),
        "input_a and input_b must have the same shape");
    TT_FATAL(
        tensor_args.input_a.memory_config() == tensor_args.input_b.memory_config(),
        "input_a and input_b must have the same memory config");
}

void ShardedAddOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}

ShardedAddOperation::spec_return_value_t ShardedAddOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    (void)tensor_args;
    // TODO: Implement compute_output_specs
    //
    // The output should mirror input_a's spec exactly (same shape, dtype, layout, memory config).
    // Hint: return tensor_args.input_a.tensor_spec();
    TT_THROW("Not implemented");
}

ShardedAddOperation::tensor_return_value_t ShardedAddOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_a.device());
}

}  // namespace ttnn::operations::onboarding

namespace ttnn::prim {

ttnn::Tensor onboarding_sharded_add(const ttnn::Tensor& input_a, const ttnn::Tensor& input_b) {
    using OperationType = ttnn::operations::onboarding::ShardedAddOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{}, OperationType::tensor_args_t{input_a, input_b});
}

}  // namespace ttnn::prim
