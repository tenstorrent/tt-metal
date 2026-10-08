// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step3_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/caller_owned_topology.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step3 {

void MorehClipGradNormStep3Operation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    auto input_tensors = tensor_args.inputs;
    for (const auto& input : input_tensors) {
        ttnn::operations::check_tensor(input, "moreh_clip_grad_norm_step3", "input");
    }

    ttnn::operations::check_tensor(tensor_args.clip_coef_clamped, "moreh_clip_grad_norm_step3", "clip_coef_clamped");
}

void MorehClipGradNormStep3Operation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

// No output
MorehClipGradNormStep3Operation::spec_return_value_t MorehClipGradNormStep3Operation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    std::vector<tt::tt_metal::TensorSpec> output_specs;
    output_specs.reserve(tensor_args.inputs.size());
    for (const auto& input : tensor_args.inputs) {
        output_specs.push_back(input.tensor_spec());
    }
    return output_specs;
}

// No output
MorehClipGradNormStep3Operation::tensor_return_value_t MorehClipGradNormStep3Operation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.inputs;
};

std::vector<tt::tt_metal::TensorTopology> MorehClipGradNormStep3Operation::compute_output_topologies(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // The inputs are scaled in place by clip_coef_clamped and returned as the outputs. Each keeps its own label
    // while the coefficient's label is compatible with it (core::caller_owned_output_topology). A coefficient
    // computed per device (any gradient sharded across the mesh makes step1/step2's norm per-device) leaves a
    // replicated gradient different on every device, and that gradient takes the union of every tensor here.
    if (tensor_args.inputs.empty()) {
        return {};
    }
    std::vector<std::reference_wrapper<const Tensor>> all_tensors(tensor_args.inputs.begin(), tensor_args.inputs.end());
    all_tensors.emplace_back(tensor_args.clip_coef_clamped);
    const auto union_topology = union_output_topology(all_tensors, tensor_args.inputs.front());
    std::vector<tt::tt_metal::TensorTopology> topologies;
    topologies.reserve(tensor_args.inputs.size());
    for (const auto& input : tensor_args.inputs) {
        topologies.push_back(ttnn::operations::core::caller_owned_output_topology(
                                 input, {&tensor_args.clip_coef_clamped}, "ttnn::moreh_clip_grad_norm (step 3)")
                                 .value_or(union_topology));
    }
    return topologies;
}
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step3

namespace ttnn::prim {
ttnn::operations::moreh::moreh_clip_grad_norm_step3::MorehClipGradNormStep3Operation::tensor_return_value_t
moreh_clip_grad_norm_step3(
    const std::vector<Tensor>& inputs,
    const Tensor& clip_coef_clamped,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_clip_grad_norm_step3::MorehClipGradNormStep3Operation;
    auto operation_attributes = OperationType::operation_attributes_t{
        memory_config.value_or(inputs.at(0).memory_config()), compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{inputs, clip_coef_clamped};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
