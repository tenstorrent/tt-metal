// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_norm_backward {

void MorehNormBackwardOperation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    check_tensor(tensor_args.input, "moreh_norm_backward", "input");
    check_tensor(tensor_args.output, "moreh_norm_backward", "output");
    check_tensor(tensor_args.output_grad, "moreh_norm_backward", "output_grad");
    check_tensor(tensor_args.input_grad, "moreh_norm_backward", "input_grad");
}

MorehNormBackwardOperation::program_factory_t MorehNormBackwardOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
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

MorehNormBackwardOperation::spec_return_value_t MorehNormBackwardOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return tensor_args.input_grad->tensor_spec();
    }
    return TensorSpec(
        tensor_args.input.logical_shape(),
        TensorLayout(
            tensor_args.input.dtype(), PageConfig(tensor_args.input.layout()), operation_attributes.memory_config));
};

MorehNormBackwardOperation::tensor_return_value_t MorehNormBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad.has_value()) {
        return tensor_args.input_grad.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::moreh::moreh_norm_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_norm_backward::MorehNormBackwardOperation::tensor_return_value_t moreh_norm_backward(
    const Tensor& input,
    const Tensor& output,
    const Tensor& output_grad,
    float p,
    const std::optional<std::variant<int64_t, ttnn::SmallVector<int64_t>>>& dim,
    bool keepdim,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_norm_backward::MorehNormBackwardOperation;
    ttnn::SmallVector<int64_t> dims = ttnn::operations::get_dim(dim, input.padded_shape().rank());
    std::sort(dims.begin(), dims.end());
    auto operation_attributes = OperationType::operation_attributes_t{
        p,
        dims,
        keepdim,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{input, output, output_grad, input_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
