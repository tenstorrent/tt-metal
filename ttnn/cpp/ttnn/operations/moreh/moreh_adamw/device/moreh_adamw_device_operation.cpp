// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adamw_device_operation.hpp"

#include <optional>

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_adamw {

MorehAdamWDeviceOperation::program_factory_t MorehAdamWDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MultiCore{};
}

void MorehAdamWDeviceOperation::validate_inputs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    check_tensor(tensor_args.param_in, "moreh_adamw", "param_in", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    check_tensor(tensor_args.grad, "moreh_adamw", "grad", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    check_tensor(tensor_args.exp_avg_in, "moreh_adamw", "exp_avg_in", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    check_tensor(tensor_args.exp_avg_sq_in, "moreh_adamw", "exp_avg_sq_in", {DataType::BFLOAT16, DataType::BFLOAT8_B});

    if (tensor_args.max_exp_avg_sq_in.has_value()) {
        check_tensor(
            tensor_args.max_exp_avg_sq_in.value(),
            "moreh_adamw",
            "max_exp_avg_sq_in",
            {DataType::BFLOAT16, DataType::BFLOAT8_B});
    }

    if (tensor_args.param_out.has_value()) {
        check_tensor(
            tensor_args.param_out.value(), "moreh_adamw", "param_out", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    }
    if (tensor_args.exp_avg_out.has_value()) {
        check_tensor(
            tensor_args.exp_avg_out.value(), "moreh_adamw", "exp_avg_out", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    }
    if (tensor_args.exp_avg_sq_out.has_value()) {
        check_tensor(
            tensor_args.exp_avg_sq_out.value(),
            "moreh_adamw",
            "exp_avg_sq_out",
            {DataType::BFLOAT16, DataType::BFLOAT8_B});
    }
    if (tensor_args.max_exp_avg_sq_out.has_value()) {
        check_tensor(
            tensor_args.max_exp_avg_sq_out.value(),
            "moreh_adamw",
            "max_exp_avg_sq_out",
            {DataType::BFLOAT16, DataType::BFLOAT8_B});
    }
}

void MorehAdamWDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

void MorehAdamWDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

MorehAdamWDeviceOperation::spec_return_value_t MorehAdamWDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_shape = tensor_args.param_in.logical_shape();
    auto dtype = tensor_args.param_in.dtype();
    auto memory_config = operation_attributes.memory_config;

    std::vector<std::optional<TensorSpec>> result;
    TensorSpec outSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));

    if (tensor_args.param_out.has_value()) {
        result.push_back(tensor_args.param_out->tensor_spec());
    } else {
        result.push_back(outSpec);
    }

    if (tensor_args.exp_avg_out.has_value()) {
        result.push_back(tensor_args.exp_avg_out->tensor_spec());
    } else {
        result.push_back(outSpec);
    }

    if (tensor_args.exp_avg_sq_out.has_value()) {
        result.push_back(tensor_args.exp_avg_sq_out->tensor_spec());
    } else {
        result.push_back(outSpec);
    }

    if (tensor_args.max_exp_avg_sq_out.has_value()) {
        result.push_back(tensor_args.max_exp_avg_sq_out->tensor_spec());
    } else if (operation_attributes.amsgrad) {
        result.push_back(outSpec);
    } else {
        result.push_back(std::nullopt);
    }

    return result;
}

MorehAdamWDeviceOperation::tensor_return_value_t MorehAdamWDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    auto device = tensor_args.param_in.device();

    tensor_return_value_t result;

    if (tensor_args.param_out.has_value()) {
        result.push_back(tensor_args.param_out.value());
    } else {
        result.push_back(create_device_tensor(*output_specs[0], device));
    }

    if (tensor_args.exp_avg_out.has_value()) {
        result.push_back(tensor_args.exp_avg_out.value());
    } else {
        result.push_back(create_device_tensor(*output_specs[1], device));
    }

    if (tensor_args.exp_avg_sq_out.has_value()) {
        result.push_back(tensor_args.exp_avg_sq_out.value());
    } else {
        result.push_back(create_device_tensor(*output_specs[2], device));
    }

    if (tensor_args.max_exp_avg_sq_out.has_value()) {
        result.push_back(tensor_args.max_exp_avg_sq_out.value());
    } else if (output_specs[3].has_value()) {
        result.push_back(create_device_tensor(*output_specs[3], device));
    } else {
        result.push_back(std::nullopt);
    }

    return result;
}

std::tuple<MorehAdamWDeviceOperation::operation_attributes_t, MorehAdamWDeviceOperation::tensor_args_t>
MorehAdamWDeviceOperation::invoke(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,

    const std::optional<float> lr,
    const std::optional<float> beta1,
    const std::optional<float> beta2,
    const std::optional<float> eps,
    const std::optional<float> weight_decay,
    const std::optional<uint32_t> step,
    const std::optional<bool> amsgrad,

    const std::optional<Tensor>& max_exp_avg_sq_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& exp_avg_out,
    const std::optional<Tensor>& exp_avg_sq_out,
    const std::optional<Tensor>& max_exp_avg_sq_out,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return {
        operation_attributes_t{
            lr.value_or(0.001f),
            beta1.value_or(0.9f),
            beta2.value_or(0.999f),
            eps.value_or(1e-8f),
            weight_decay.value_or(1e-2f),
            step.value_or(0),
            amsgrad.value_or(false),
            memory_config.value_or(param_in.memory_config()),
            init_device_compute_kernel_config(param_in.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        tensor_args_t{
            param_in,
            grad,
            exp_avg_in,
            exp_avg_sq_in,
            max_exp_avg_sq_in,
            param_out,
            exp_avg_out,
            exp_avg_sq_out,
            max_exp_avg_sq_out}};
}

tt::stl::hash::hash_t MorehAdamWDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto operation_attributes_without_step_and_lr = operation_attributes;
    operation_attributes_without_step_and_lr.step = 0;
    operation_attributes_without_step_and_lr.lr = 0.0f;
    return tt::stl::hash::hash_objects_with_default_seed(operation_attributes_without_step_and_lr, tensor_args);
}
}  // namespace ttnn::operations::moreh::moreh_adamw
