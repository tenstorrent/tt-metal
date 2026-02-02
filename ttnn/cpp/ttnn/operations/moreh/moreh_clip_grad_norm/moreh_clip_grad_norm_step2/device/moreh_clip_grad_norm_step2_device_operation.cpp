// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step2_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step2 {

void MorehClipGradNormStep2Operation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    check_tensor(tensor_args.tmp_pow_sum, "moreh_clip_grad_norm_step2", "tmp_pow_sum");

    if (tensor_args.total_norm.has_value()) {
        check_tensor(tensor_args.total_norm, "moreh_clip_grad_norm_step2", "total_norm");
    }
};

MorehClipGradNormStep2Operation::program_factory_t MorehClipGradNormStep2Operation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
};

void MorehClipGradNormStep2Operation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehClipGradNormStep2Operation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehClipGradNormStep2Operation::spec_return_value_t MorehClipGradNormStep2Operation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.total_norm.has_value()) {
        return tensor_args.total_norm->tensor_spec();
    }

    // output total_norm 1 element
    return TensorSpec(
        Shape{1, 1},
        TensorLayout(tensor_args.tmp_pow_sum.dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
};

MorehClipGradNormStep2Operation::tensor_return_value_t MorehClipGradNormStep2Operation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.total_norm.has_value()) {
        return tensor_args.total_norm.value();
    }

    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.tmp_pow_sum.device());
};

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step2

namespace ttnn::prim {
ttnn::operations::moreh::moreh_clip_grad_norm_step2::MorehClipGradNormStep2Operation::tensor_return_value_t
moreh_clip_grad_norm_step2(
    const Tensor& tmp_pow_sum,
    float norm_type,
    const std::optional<Tensor>& total_norm,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_clip_grad_norm_step2::MorehClipGradNormStep2Operation;
    auto operation_attributes = OperationType::operation_attributes_t{
        norm_type, memory_config.value_or(tmp_pow_sum.memory_config()), compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{tmp_pow_sum, total_norm};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
