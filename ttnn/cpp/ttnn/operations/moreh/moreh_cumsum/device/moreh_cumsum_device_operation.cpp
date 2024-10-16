// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {
void MorehCumsumDeviceOperation::validate_inputs(const operation_attributes_t& operation_attributes,
                                                 const tensor_args_t& tensor_args) {
    const auto dim = operation_attributes.dim;
    TT_FATAL((dim >= 0 && dim <= 3), "dim should be 0 - 3, but got: {}", dim);
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    if (!output.has_value()) {
        return;
    }

    const auto input_shape = input.get_shape();
    const auto output_shape = output.value().get_shape();
    const auto input_shape_wo_padding = input_shape.value.without_padding();
    const auto output_shape_wo_padding = output_shape.value.without_padding();

    for (int i = 0; i < input_shape.rank(); ++i) {
        TT_FATAL(input_shape[i] == output_shape[i],
                 "Input shape must match output shape. Received input_shape = {} and output_shape = {}.",
                 input_shape[i],
                 output_shape[i]);
        TT_FATAL(input_shape_wo_padding[i] == output_shape_wo_padding[i],
                 "Input and output shapes (excluding padding) must be equal. Received input_shape_wo_padding = {} and "
                 "output_shape_wo_padding = {}.",
                 input_shape_wo_padding[i],
                 output_shape_wo_padding[i]);
    }
}

MorehCumsumDeviceOperation::program_factory_t MorehCumsumDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehCumsumDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t& operation_attributes,
                                                                const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehCumsumDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t& operation_attributes,
                                                               const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehCumsumDeviceOperation::shape_return_value_t MorehCumsumDeviceOperation::compute_output_shapes(
    const operation_attributes_t&,
    const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    auto output_shape = input.get_shape();
    return output_shape;
}

MorehCumsumDeviceOperation::tensor_return_value_t MorehCumsumDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;
    if (output.has_value()) {
        return output.value();
    }

    auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    return create_device_tensor(
        output_shape, input.tensor_attributes->dtype, input.tensor_attributes->layout, input.device());
}

std::tuple<MorehCumsumDeviceOperation::operation_attributes_t, MorehCumsumDeviceOperation::tensor_args_t>
MorehCumsumDeviceOperation::invoke(const Tensor& input,
                                   const int64_t dim,
                                   const std::optional<Tensor>& output,
                                   const bool flip,
                                   const std::optional<MemoryConfig>& memory_config,
                                   const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {operation_attributes_t{
                dim,
                flip,
                memory_config.value_or(input.memory_config()),
                init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
            tensor_args_t{input, output}};
}
}  // namespace ttnn::operations::moreh::moreh_cumsum
