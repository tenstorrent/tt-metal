// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_adam_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <cstdint>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_adam {
void MorehAdamOperation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& params_in = tensor_args.param_in;
    const auto& grad = tensor_args.grad;
    const auto& exp_avg_in = tensor_args.exp_avg_in;
    const auto& exp_avg_sq_in = tensor_args.exp_avg_sq_in;

    check_tensor(params_in, "moreh_adam", "params_in");
    check_tensor(grad, "moreh_adam", "grad");
    check_tensor(exp_avg_in, "moreh_adam", "exp_avg_in");
    check_tensor(exp_avg_sq_in, "moreh_adam", "exp_avg_sq_in");

    if (tensor_args.max_exp_avg_sq_in) {
        check_tensor(*tensor_args.max_exp_avg_sq_in, "moreh_adam", "max_exp_avg_sq_in");
    }

    const auto& params_out = tensor_args.output_tensors.at(0);

    if (params_out.has_value()) {
        check_tensor(params_out.value(), "moreh_adam", "params_out");
    }

    if (tensor_args.output_tensors.at(1).has_value()) {
        check_tensor(tensor_args.output_tensors.at(1).value(), "moreh_adam", "exp_avg_out");
    }

    if (tensor_args.output_tensors.at(2).has_value()) {
        check_tensor(tensor_args.output_tensors.at(2).value(), "moreh_adam", "exp_avg_sq_out");
    }

    if (tensor_args.output_tensors.at(3).has_value()) {
        check_tensor(tensor_args.output_tensors.at(3).value(), "moreh_adam", "max_exp_avg_sq_out");
    }
}

MorehAdamOperation::program_factory_t MorehAdamOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // For now we litteraly don't care and return a single factory. Whatever
    return ProgramFactory{};
}

void MorehAdamOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehAdamOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehAdamOperation::spec_return_value_t MorehAdamOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_shape = tensor_args.param_in.logical_shape();
    auto dtype = tensor_args.param_in.dtype();

    std::vector<std::optional<TensorSpec>> ret;
    TensorSpec out_spec(
        output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), operation_attributes.memory_config));
    for (int idx = 0; idx < 3; idx++) {
        if (tensor_args.output_tensors.at(idx).has_value()) {
            ret.push_back(tensor_args.output_tensors.at(idx)->tensor_spec());
        } else {
            ret.push_back(out_spec);
        }
    }
    if (tensor_args.output_tensors.at(3).has_value()) {
        ret.push_back(tensor_args.output_tensors.at(3)->tensor_spec());
    } else if (operation_attributes.amsgrad) {
        ret.push_back(out_spec);
    } else {
        ret.push_back(std::nullopt);
    }

    return ret;
}

MorehAdamOperation::tensor_return_value_t MorehAdamOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.param_in.device();

    std::vector<std::optional<Tensor>> ret;
    auto memory_config = operation_attributes.memory_config;

    for (size_t idx = 0; idx < output_specs.size(); idx++) {
        if (tensor_args.output_tensors.at(idx).has_value()) {
            ret.push_back(tensor_args.output_tensors.at(idx));
        } else if (output_specs[idx].has_value()) {
            ret.push_back(create_device_tensor(*output_specs[idx], device));
        } else {
            ret.push_back(std::nullopt);
        }
    }

    return ret;
}

auto MorehAdamOperation::compute_program_hash(
    const MorehAdamOperation::operation_attributes_t& operation_attributes,
    const MorehAdamOperation::tensor_args_t& tensor_args) -> tt::stl::hash::hash_t {
    auto operation_attributes_without_step_and_lr = operation_attributes;
    operation_attributes_without_step_and_lr.step = 0;
    operation_attributes_without_step_and_lr.lr = 0.0f;
    return tt::stl::hash::hash_objects_with_default_seed(operation_attributes_without_step_and_lr, tensor_args);
}
}  // namespace ttnn::operations::moreh::moreh_adam

namespace ttnn::prim {
ttnn::operations::moreh::moreh_adam::MorehAdamOperation::tensor_return_value_t moreh_adam(
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
    const std::optional<const Tensor>& max_exp_avg_sq_in,
    const std::optional<const Tensor> param_out,
    const std::optional<const Tensor> exp_avg_out,
    const std::optional<const Tensor> exp_avg_sq_out,
    const std::optional<const Tensor> max_exp_avg_sq_out,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_adam::MorehAdamOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        lr.value_or(0.001f),
        beta1.value_or(0.9f),
        beta2.value_or(0.999f),
        eps.value_or(1e-8f),
        weight_decay.value_or(0.0f),
        step.value_or(0),
        amsgrad.value_or(false),
        memory_config.value_or(param_in.memory_config()),
        init_device_compute_kernel_config(param_in.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{
        param_in,
        grad,
        exp_avg_in,
        exp_avg_sq_in,
        max_exp_avg_sq_in,
        {param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out}};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
