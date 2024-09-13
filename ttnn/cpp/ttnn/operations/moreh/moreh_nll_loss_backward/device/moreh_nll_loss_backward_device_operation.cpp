// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

MorehNllLossBackwardDeviceOperation::program_factory_t MorehNllLossBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return Factory{};
}

void MorehNllLossBackwardDeviceOperation::validate_inputs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto& target_tensor = tensor_args.target_tensor;
    auto& output_grad_tensor = tensor_args.output_grad_tensor;
    auto& weight_tensor = tensor_args.weight_tensor;
    auto& divisor_tensor = tensor_args.divisor_tensor;
    auto& input_grad_tensor = tensor_args.input_grad_tensor;

    TT_FATAL(target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_FATAL(target_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_FATAL((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_FATAL(target_tensor.get_dtype() == DataType::INT32, "Invalid target_tensor dtype {}", target_tensor.get_dtype());

    TT_FATAL(output_grad_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_FATAL(output_grad_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_FATAL((output_grad_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_FATAL(
        output_grad_tensor.get_dtype() == DataType::BFLOAT16,
        "Invalid output_grad_tensor dtype {}",
        output_grad_tensor.get_dtype());

    if (input_grad_tensor.has_value()) {
        TT_FATAL(
            input_grad_tensor.value().storage_type() == StorageType::DEVICE,
            "Operands to nll_loss need to be on device!");
        TT_FATAL(
            input_grad_tensor.value().buffer() != nullptr,
            "Operands to nll_loss need to be allocated in buffers on device!");
        TT_FATAL((input_grad_tensor.value().get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
        TT_FATAL(
            input_grad_tensor.value().get_dtype() == DataType::BFLOAT16,
            "Invalid input_grad_tensor dtype {}",
            input_grad_tensor.value().get_dtype());
    }

    if (weight_tensor.has_value()) {
        TT_FATAL(
            weight_tensor.value().storage_type() == StorageType::DEVICE,
            "weight_tensor to nll_loss need to be on device!");
        TT_FATAL(
            weight_tensor.value().buffer() != nullptr,
            "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL((weight_tensor.value().get_layout() == Layout::TILE), "weight_tensor to nll_loss must be in tilized");
        TT_FATAL(
            weight_tensor.value().get_dtype() == DataType::BFLOAT16,
            "Invalid weight_tensor dtype {}",
            weight_tensor.value().get_dtype());
    }

    if (divisor_tensor.has_value()) {
        TT_FATAL(
            divisor_tensor.value().storage_type() == StorageType::DEVICE,
            "divisor_tensor to nll_loss need to be on device!");
        TT_FATAL(
            divisor_tensor.value().buffer() != nullptr,
            "divisor_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL((divisor_tensor.value().get_layout() == Layout::TILE), "divisor_tensor to nll_loss must be tilized");
        TT_FATAL(
            divisor_tensor.value().get_dtype() == DataType::BFLOAT16,
            "Invalid divisor_tensor dtype {}",
            divisor_tensor.value().get_dtype());
    }
}

void MorehNllLossBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

void MorehNllLossBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

MorehNllLossBackwardDeviceOperation::shape_return_value_t MorehNllLossBackwardDeviceOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // To calculate the output shape, we need the channel_size. However, the required tensors, target and output_grad,
    // do not contain the channel_size information.
    TT_FATAL(false, "moreh_nll_loss_backward not support create output tensors.");
    return tensor_args.target_tensor.get_shape();
}

MorehNllLossBackwardDeviceOperation::tensor_return_value_t MorehNllLossBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad_tensor.has_value()) {
        return {tensor_args.input_grad_tensor.value()};
    }

    auto output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.target_tensor.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.target_tensor.device();
    return create_device_tensor(output_shapes, dtype, layout, device, operation_attributes.memory_config);
}

std::tuple<
    MorehNllLossBackwardDeviceOperation::operation_attributes_t,
    MorehNllLossBackwardDeviceOperation::tensor_args_t>
MorehNllLossBackwardDeviceOperation::invoke(
    const Tensor& target_tensor,
    const Tensor& output_grad_tensor,
    const bool reduction_mean,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> input_grad_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const int32_t ignore_index,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return {
        operation_attributes_t{
            reduction_mean,
            ignore_index < 0 ? std::numeric_limits<uint32_t>::max() : ignore_index,
            memory_config.value_or(target_tensor.memory_config()),
            compute_kernel_config},
        tensor_args_t{target_tensor, output_grad_tensor, weight_tensor, divisor_tensor, input_grad_tensor}};
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward
