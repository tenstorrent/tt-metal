// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_step1_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step1 {

void MorehClipGradNormStep1Operation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensors = tensor_args.inputs;
    for (const auto& input : input_tensors) {
        ttnn::operations::check_tensor(input, "moreh_clip_grad_norm_step1", "input");
    }

    ttnn::operations::check_tensor(tensor_args.tmp_pow_sum, "moreh_clip_grad_norm_step1", "tmp_pow_sum");
};

MorehClipGradNormStep1Operation::program_factory_t MorehClipGradNormStep1Operation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
};

void MorehClipGradNormStep1Operation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehClipGradNormStep1Operation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehClipGradNormStep1Operation::spec_return_value_t MorehClipGradNormStep1Operation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.tmp_pow_sum.get_tensor_spec();
};

MorehClipGradNormStep1Operation::tensor_return_value_t MorehClipGradNormStep1Operation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.tmp_pow_sum;
};

std::tuple<MorehClipGradNormStep1Operation::operation_attributes_t, MorehClipGradNormStep1Operation::tensor_args_t>
MorehClipGradNormStep1Operation::invoke(
    const std::vector<Tensor>& inputs,
    const float norm_type,
    const uint32_t tile_offset_of_tmp_pow_sum,
    const Tensor& tmp_pow_sum,
    const std::optional<MemoryConfig>& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    return {
        operation_attributes_t{
            norm_type,
            tile_offset_of_tmp_pow_sum,
            memory_config.value_or(inputs.at(0).memory_config()),
            compute_kernel_config},
        tensor_args_t{inputs, tmp_pow_sum}};
};
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step1
