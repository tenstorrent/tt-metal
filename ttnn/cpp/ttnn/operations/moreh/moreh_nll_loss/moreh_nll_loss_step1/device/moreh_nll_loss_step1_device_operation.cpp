// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_step1_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_step1 {

MorehNllLossStep1DeviceOperation::program_factory_t MorehNllLossStep1DeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return Factory{};
}

void MorehNllLossStep1DeviceOperation::validate_inputs(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    const auto& target_tensor = tensor_args.target_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;

    TT_FATAL(target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_FATAL(target_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_FATAL((target_tensor.layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");

    if (weight_tensor.has_value()) {
        TT_FATAL(
            weight_tensor.value().storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
        TT_FATAL(
            weight_tensor.value().buffer() != nullptr,
            "Operands to nll_loss need to be allocated in buffers on device!");
        TT_FATAL(weight_tensor.value().dtype() == DataType::BFLOAT16, "weigth tensor dtype must be bfloat16");
    }
}

void MorehNllLossStep1DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

void MorehNllLossStep1DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

MorehNllLossStep1DeviceOperation::spec_return_value_t MorehNllLossStep1DeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& target_tensor = tensor_args.target_tensor;
    return TensorSpec(
        target_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype, tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.memory_config));
}

MorehNllLossStep1DeviceOperation::tensor_return_value_t MorehNllLossStep1DeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.target_tensor.device());
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step1

namespace ttnn::prim {
ttnn::operations::moreh::moreh_nll_loss_step1::MorehNllLossStep1DeviceOperation::tensor_return_value_t
moreh_nll_loss_step1(
    const Tensor& target_tensor,
    const std::optional<Tensor>& weight_tensor,
    int32_t ignore_index,
    const std::string& reduction,
    DataType dtype,
    uint32_t channel_size,
    const std::optional<MemoryConfig>& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_nll_loss_step1::MorehNllLossStep1DeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        reduction,
        ignore_index < 0 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(ignore_index),
        dtype,
        channel_size,
        memory_config.value_or(target_tensor.memory_config()),
        compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{target_tensor, weight_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
