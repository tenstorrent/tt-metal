// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_step1_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_step1 {

MorehNllLossStep1DeviceOperation::program_factory_t MorehNllLossStep1DeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    return Factory{};
}

void MorehNllLossStep1DeviceOperation::validate_inputs(const operation_attributes_t& attributes,
                                                       const tensor_args_t& tensor_args) {
    auto& target_tensor = tensor_args.target_tensor;
    auto& weight_tensor = tensor_args.weight_tensor;

    TT_FATAL(target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_FATAL(target_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_FATAL((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");

    if (weight_tensor.has_value()) {
        TT_FATAL(weight_tensor.value().storage_type() == StorageType::DEVICE,
                 "Operands to nll_loss need to be on device!");
        TT_FATAL(weight_tensor.value().buffer() != nullptr,
                 "Operands to nll_loss need to be allocated in buffers on device!");
        TT_FATAL(weight_tensor.value().get_dtype() == DataType::BFLOAT16, "weigth tensor dtype must be bfloat16");
    }
}

void MorehNllLossStep1DeviceOperation::validate_on_program_cache_miss(const operation_attributes_t& attributes,
                                                                      const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

void MorehNllLossStep1DeviceOperation::validate_on_program_cache_hit(const operation_attributes_t& attributes,
                                                                     const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

MorehNllLossStep1DeviceOperation::shape_return_value_t MorehNllLossStep1DeviceOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    const auto& target_tensor = tensor_args.target_tensor;
    auto target_shape = target_tensor.get_shape();

    return target_shape;
}

MorehNllLossStep1DeviceOperation::tensor_return_value_t MorehNllLossStep1DeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    const auto& target_tensor = tensor_args.target_tensor;
    auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    Layout layout{Layout::TILE};
    auto device = tensor_args.target_tensor.device();

    return create_device_tensor(
        output_shape, operation_attributes.dtype, layout, device, operation_attributes.memory_config);
}

std::tuple<MorehNllLossStep1DeviceOperation::operation_attributes_t, MorehNllLossStep1DeviceOperation::tensor_args_t>
MorehNllLossStep1DeviceOperation::invoke(const Tensor& target_tensor,
                                         const std::optional<Tensor>& weight_tensor,
                                         const int32_t ignore_index,
                                         const std::string reduction,
                                         const DataType dtype,
                                         const uint32_t channel_size,
                                         const std::optional<MemoryConfig>& memory_config,
                                         const DeviceComputeKernelConfig& compute_kernel_config) {
    return {operation_attributes_t{reduction,
                                   ignore_index < 0 ? std::numeric_limits<uint32_t>::max() : ignore_index,
                                   dtype,
                                   channel_size,
                                   memory_config.value_or(target_tensor.memory_config()),
                                   compute_kernel_config},
            tensor_args_t{target_tensor, weight_tensor}};
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step1
