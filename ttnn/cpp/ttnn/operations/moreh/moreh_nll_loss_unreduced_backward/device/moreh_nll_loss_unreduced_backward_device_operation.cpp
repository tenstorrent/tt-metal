// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_unreduced_backward_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward {

MorehNllLossUnreducedBackwardDeviceOperation::program_factory_t
MorehNllLossUnreducedBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return Factory{};
}

void MorehNllLossUnreducedBackwardDeviceOperation::validate_inputs(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    const auto& target_tensor = tensor_args.target_tensor;
    const auto& output_grad_tensor = tensor_args.output_grad_tensor;

    const auto& weight_tensor = tensor_args.weight_tensor;

    const auto& input_grad_tensor = tensor_args.input_grad_tensor;

    TT_FATAL(
        target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss_unreduced need to be on device!");
    TT_FATAL(
        target_tensor.buffer() != nullptr, "Operands to nll_loss_unreduced need to be allocated in buffers on device!");
    TT_FATAL((target_tensor.layout() == Layout::TILE), "target_tensor to nll_loss_unreduced must be tilized");
    TT_FATAL(target_tensor.dtype() == DataType::INT32, "Invalid target_tensor dtype {}", target_tensor.dtype());

    TT_FATAL(
        output_grad_tensor.storage_type() == StorageType::DEVICE,
        "Operands to nll_loss_unreduced need to be on device!");
    TT_FATAL(
        output_grad_tensor.buffer() != nullptr,
        "Operands to nll_loss_unreduced need to be allocated in buffers on device!");
    TT_FATAL((output_grad_tensor.layout() == Layout::TILE), "target_tensor to nll_loss_unreduced must be tilized");
    TT_FATAL(
        output_grad_tensor.dtype() == DataType::BFLOAT16,
        "Invalid output_grad_tensor dtype {}",
        output_grad_tensor.dtype());

    if (input_grad_tensor.has_value()) {
        TT_FATAL(
            input_grad_tensor.value().storage_type() == StorageType::DEVICE,
            "Operands to nll_loss need to be on device!");
        TT_FATAL(
            input_grad_tensor.value().buffer() != nullptr,
            "Operands to nll_loss need to be allocated in buffers on device!");
        TT_FATAL(
            (input_grad_tensor.value().layout() == Layout::TILE),
            "target_tensor to nll_loss_unreduced must be tilized");
        TT_FATAL(
            input_grad_tensor.value().dtype() == DataType::BFLOAT16,
            "Invalid input_grad_tensor dtype {}",
            input_grad_tensor.value().dtype());
    }

    if (weight_tensor.has_value()) {
        TT_FATAL(
            weight_tensor.value().storage_type() == StorageType::DEVICE,
            "weight_tensor to nll_loss need to be on device!");
        TT_FATAL(
            weight_tensor.value().buffer() != nullptr,
            "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL(
            (weight_tensor.value().layout() == Layout::TILE), "weight_tensor to nll_loss_unreduced must be in tilized");
        TT_FATAL(
            weight_tensor.value().dtype() == DataType::BFLOAT16,
            "Invalid weight_tensor dtype {}",
            weight_tensor.value().dtype());
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

MorehNllLossUnreducedBackwardDeviceOperation::spec_return_value_t
MorehNllLossUnreducedBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad_tensor.has_value()) {
        return {tensor_args.input_grad_tensor->tensor_spec()};
    }
    // To calculate the output shape, we need the channel_size. However, the required tensors, target and output_grad,
    // do not contain the channel_size information.
    TT_FATAL(false, "moreh_nll_loss_unreduced_backward not support creating output tensors.");
}

MorehNllLossUnreducedBackwardDeviceOperation::tensor_return_value_t
MorehNllLossUnreducedBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_grad_tensor.has_value()) {
        return {tensor_args.input_grad_tensor.value()};
    }

    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.target_tensor.device());
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_nll_loss_unreduced_backward::MorehNllLossUnreducedBackwardDeviceOperation::
    tensor_return_value_t
    moreh_nll_loss_unreduced_backward(
        const Tensor& target_tensor,
        const Tensor& output_grad_tensor,
        const std::optional<Tensor>& weight_tensor,
        const std::optional<Tensor>& input_grad_tensor,
        int32_t ignore_index,
        const std::optional<ttnn::MemoryConfig>& memory_config,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType =
        ttnn::operations::moreh::moreh_nll_loss_unreduced_backward::MorehNllLossUnreducedBackwardDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        ignore_index < 0 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(ignore_index),
        memory_config.value_or(target_tensor.memory_config()),
        init_device_compute_kernel_config(target_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{target_tensor, output_grad_tensor, weight_tensor, input_grad_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
