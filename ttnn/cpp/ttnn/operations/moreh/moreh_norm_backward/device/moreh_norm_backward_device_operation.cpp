// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward_device_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_norm_backward {

void MorehNormBackwardOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    tt::operations::primary::check_tensor(tensor_args.input, "moreh_norm_backward", "input");
    tt::operations::primary::check_tensor(tensor_args.output, "moreh_norm_backward", "output");
    tt::operations::primary::check_tensor(tensor_args.output_grad, "moreh_norm_backward", "output_grad");
    tt::operations::primary::check_tensor(tensor_args.input_grad, "moreh_norm_backward", "input_grad");
}

MorehNormBackwardOperation::program_factory_t MorehNormBackwardOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehNormBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehNormBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehNormBackwardOperation::shape_return_value_t MorehNormBackwardOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input.get_shape();
};

MorehNormBackwardOperation::tensor_return_value_t MorehNormBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value())
        return tensor_args.input_grad.value();
    const auto& input = tensor_args.input;
    return create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        input.get_dtype(),
        input.get_layout(),
        input.device(),
        operation_attributes.memory_config);
}

std::tuple<MorehNormBackwardOperation::operation_attributes_t, MorehNormBackwardOperation::tensor_args_t>
MorehNormBackwardOperation::invoke(
    const Tensor& input,
    const Tensor& output,
    const Tensor& output_grad,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    bool keepdim,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    std::vector<int64_t> dims = tt::operations::primary::get_dim(dim, input.get_legacy_shape().rank());
    std::sort(dims.begin(), dims.end());
    return {
        operation_attributes_t{
            p,
            dims,
            keepdim,
            memory_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        tensor_args_t{
            input,
            output,
            output_grad,
            input_grad,
        },
    };
}
}  // namespace ttnn::operations::moreh::moreh_norm_backward
