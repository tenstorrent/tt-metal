// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sgd_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_sgd {
void MorehSgdOperation::validate_inputs(const operation_attributes_t& operation_attributes,
                                        const tensor_args_t& tensor_args) {
    auto& params_in = tensor_args.param_in;
    auto& grad = tensor_args.grad;

    tt::operations::primary::check_tensor(params_in, "moreh_sgd", "params_in");
    tt::operations::primary::check_tensor(grad, "moreh_sgd", "grad");

    if (tensor_args.momentum_buffer_in) {
        tt::operations::primary::check_tensor(*tensor_args.momentum_buffer_in, "moreh_sgd", "momentum_buffer_in");
    }

    if (tensor_args.param_out.has_value()) {
        tt::operations::primary::check_tensor(tensor_args.param_out.value(), "moreh_sgd", "param_out");
    }

    if (tensor_args.momentum_buffer_out.has_value()) {
        tt::operations::primary::check_tensor(
            tensor_args.momentum_buffer_out.value(), "moreh_sgd", "momentum_buffer_out");
    }
}

MorehSgdOperation::program_factory_t MorehSgdOperation::select_program_factory(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    return ProgramFactory{};
};

void MorehSgdOperation::validate_on_program_cache_miss(const operation_attributes_t& operation_attributes,
                                                       const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehSgdOperation::validate_on_program_cache_hit(const operation_attributes_t& operation_attributes,
                                                      const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehSgdOperation::shape_return_value_t MorehSgdOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    auto input_tensor_shape = tensor_args.param_in.get_shape();

    return {input_tensor_shape, input_tensor_shape};
};

MorehSgdOperation::tensor_return_value_t MorehSgdOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    const auto& output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.param_in.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.param_in.device();

    std::vector<std::optional<Tensor>> ret;

    if (tensor_args.param_out.has_value()) {
        ret.push_back(tensor_args.param_out.value());
    } else {
        ret.push_back(create_device_tensor(
            output_shapes.at(0).value(), dtype, layout, device, operation_attributes.param_out_memory_config));
    }

    if (tensor_args.momentum_buffer_out.has_value()) {
        ret.push_back(tensor_args.momentum_buffer_out.value());
    } else if (operation_attributes.momentum != 0.0f) {
        ret.push_back(create_device_tensor(output_shapes.at(1).value(),
                                           dtype,
                                           layout,
                                           device,
                                           operation_attributes.momentum_buffer_out_memory_config));
    } else {
        ret.push_back(std::nullopt);
    }

    return std::move(ret);
}

std::tuple<MorehSgdOperation::operation_attributes_t, MorehSgdOperation::tensor_args_t> MorehSgdOperation::invoke(
    const Tensor& param_in,
    const Tensor& grad,
    const std::optional<Tensor>& momentum_buffer_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const std::optional<MemoryConfig>& param_out_memory_config,
    const std::optional<MemoryConfig>& momentum_buffer_out_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {operation_attributes_t{lr,
                                   momentum,
                                   dampening,
                                   weight_decay,
                                   nesterov,
                                   momentum_initialized,
                                   param_out_memory_config.value_or(param_in.memory_config()),
                                   momentum_buffer_out_memory_config.value_or(param_in.memory_config()),
                                   init_device_compute_kernel_config(
                                       param_in.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},

            tensor_args_t{param_in, grad, momentum_buffer_in, param_out, momentum_buffer_out}};
}
}  // namespace ttnn::operations::moreh::moreh_sgd
