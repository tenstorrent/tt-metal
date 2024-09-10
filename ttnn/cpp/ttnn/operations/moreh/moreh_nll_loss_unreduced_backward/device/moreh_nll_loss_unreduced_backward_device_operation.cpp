// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_unreduced_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward {

MorehNllLossUnreducedBackwardDeviceOperation::program_factory_t
MorehNllLossUnreducedBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return Factory{};
}

void MorehNllLossUnreducedBackwardDeviceOperation::validate_inputs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& target_tensor = tensor_args.target_tensor;
    const auto& output_grad_tensor = tensor_args.output_grad_tensor;

    const auto& weight_tensor = tensor_args.weight_tensor;

    const auto& input_grad_tensor = tensor_args.input_grad_tensor;

    TT_FATAL(
        target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss_unreduced need to be on device!");
    TT_FATAL(
        target_tensor.buffer() != nullptr, "Operands to nll_loss_unreduced need to be allocated in buffers on device!");
    TT_FATAL((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss_unreduced must be tilized");
    TT_FATAL(target_tensor.get_dtype() == DataType::INT32);

    TT_FATAL(
        output_grad_tensor.storage_type() == StorageType::DEVICE,
        "Operands to nll_loss_unreduced need to be on device!");
    TT_FATAL(
        output_grad_tensor.buffer() != nullptr,
        "Operands to nll_loss_unreduced need to be allocated in buffers on device!");
    TT_FATAL((output_grad_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss_unreduced must be tilized");
    TT_FATAL(output_grad_tensor.get_dtype() == DataType::BFLOAT16);

    if (input_grad_tensor.has_value()) {
        TT_FATAL(
            input_grad_tensor.value().storage_type() == StorageType::DEVICE,
            "Operands to nll_loss need to be on device!");
        TT_FATAL(
            input_grad_tensor.value().buffer() != nullptr,
            "Operands to nll_loss need to be allocated in buffers on device!");
        TT_FATAL(
            (input_grad_tensor.value().get_layout() == Layout::TILE),
            "target_tensor to nll_loss_unreduced must be tilized");
        TT_FATAL(input_grad_tensor.value().get_dtype() == DataType::BFLOAT16);
    }

    if (weight_tensor.has_value()) {
        TT_FATAL(
            weight_tensor.value().storage_type() == StorageType::DEVICE,
            "weight_tensor to nll_loss need to be on device!");
        TT_FATAL(
            weight_tensor.value().buffer() != nullptr,
            "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL(
            (weight_tensor.value().get_layout() == Layout::TILE),
            "weight_tensor to nll_loss_unreduced must be in tilized");
        TT_FATAL(weight_tensor.value().get_dtype() == DataType::BFLOAT16);
    }
}

void MorehNllLossUnreducedBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

void MorehNllLossUnreducedBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

MorehNllLossUnreducedBackwardDeviceOperation::shape_return_value_t
MorehNllLossUnreducedBackwardDeviceOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // To calculate the output shape, we need the channel_size. However, the required tensors, target and output_grad,
    // do not contain the channel_size information.
    TT_FATAL(false, "moreh_nll_loss_unreduced_backward not support create output tensors.");
    return {tensor_args.target_tensor.get_shape()};
}

MorehNllLossUnreducedBackwardDeviceOperation::tensor_return_value_t
MorehNllLossUnreducedBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad_tensor.has_value()) {
        return {tensor_args.input_grad_tensor.value()};
    }

    auto output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.target_tensor.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.target_tensor.device();
    return create_device_tensor(output_shapes.at(1), dtype, layout, device, operation_attributes.memory_config);
}

std::tuple<
    MorehNllLossUnreducedBackwardDeviceOperation::operation_attributes_t,
    MorehNllLossUnreducedBackwardDeviceOperation::tensor_args_t>
MorehNllLossUnreducedBackwardDeviceOperation::invoke(
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const Tensor& output_grad_tensor,
    const std::optional<const Tensor> input_grad_tensor,
    const int32_t ignore_index,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return {
        operation_attributes_t{
            ignore_index, memory_config.value_or(output_grad_tensor.memory_config()), compute_kernel_config},
        tensor_args_t{target_tensor, weight_tensor, output_grad_tensor, input_grad_tensor}};
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward
