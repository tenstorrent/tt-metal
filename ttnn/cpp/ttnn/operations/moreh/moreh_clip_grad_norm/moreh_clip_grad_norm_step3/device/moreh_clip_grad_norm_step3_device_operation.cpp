// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step3_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step3 {

void MorehClipGradNormStep3Operation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensors = tensor_args.inputs;
    for (const auto& input : input_tensors) {
        ttnn::operations::check_tensor(input, "moreh_clip_grad_norm_step3", "input");
    }

    ttnn::operations::check_tensor(tensor_args.clip_coef_clamped, "moreh_clip_grad_norm_step3", "clip_coef_clamped");
};

MorehClipGradNormStep3Operation::program_factory_t MorehClipGradNormStep3Operation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
};

void MorehClipGradNormStep3Operation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehClipGradNormStep3Operation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

// No output
MorehClipGradNormStep3Operation::spec_return_value_t MorehClipGradNormStep3Operation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    std::vector<TensorSpec> output_specs;
    output_specs.reserve(tensor_args.inputs.size());
    for (const auto& input : tensor_args.inputs) {
        output_specs.push_back(input.tensor_spec());
    }
    return output_specs;
};

// No output
MorehClipGradNormStep3Operation::tensor_return_value_t MorehClipGradNormStep3Operation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.inputs;
};

std::tuple<MorehClipGradNormStep3Operation::operation_attributes_t, MorehClipGradNormStep3Operation::tensor_args_t>
MorehClipGradNormStep3Operation::invoke(
    const std::vector<Tensor>& inputs,
    const Tensor& clip_coef_clamped,
    const std::optional<MemoryConfig>& memory_config,
    const DeviceComputeKernelConfig compute_kernel_config) {
    return {
        operation_attributes_t{memory_config.value_or(inputs.at(0).memory_config()), compute_kernel_config},
        tensor_args_t{inputs, clip_coef_clamped}};
};
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step3
