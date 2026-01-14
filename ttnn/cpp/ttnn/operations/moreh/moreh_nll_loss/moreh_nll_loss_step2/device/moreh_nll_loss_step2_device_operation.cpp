// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_step2_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_nll_loss_step2 {

MorehNllLossStep2DeviceOperation::program_factory_t MorehNllLossStep2DeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return Factory{};
}

void MorehNllLossStep2DeviceOperation::validate_inputs(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& target_tensor = tensor_args.target_tensor;
    const std::optional<Tensor>& weight_tensor = tensor_args.weight_tensor;
    const std::optional<Tensor>& divisor_tensor = tensor_args.divisor_tensor;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "intput_tensor to nll_loss need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "intput_tensor to nll_loss need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.layout() == Layout::TILE), "intput_tensor to nll_loss must be tilized");
    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "input tensor type must be bfloat16");

    TT_FATAL(target_tensor.storage_type() == StorageType::DEVICE, "target_tensor to nll_loss need to be on device!");
    TT_FATAL(target_tensor.buffer() != nullptr, "target_tensor to nll_loss need to be allocated in buffers on device!");
    TT_FATAL((target_tensor.layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_FATAL(target_tensor.dtype() == DataType::INT32, "target tensor type must be int32");

    if (weight_tensor.has_value()) {
        TT_FATAL(
            weight_tensor.value().storage_type() == StorageType::DEVICE,
            "weight_tensor to nll_loss need to be on device!");
        TT_FATAL(
            weight_tensor.value().buffer() != nullptr,
            "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL(
            (weight_tensor.value().layout() == Layout::TILE), "weight_tensor to nll_loss must be in row major layout");
        TT_FATAL(weight_tensor.value().dtype() == DataType::BFLOAT16, "weight tensor type must be bfloat16");
    }

    if (divisor_tensor.has_value()) {
        TT_FATAL(
            divisor_tensor.value().storage_type() == StorageType::DEVICE,
            "divisor_tensor to nll_loss need to be on device!");
        TT_FATAL(
            divisor_tensor.value().buffer() != nullptr,
            "divisor_tensor to nll_loss need to be allocated in buffers on device!");
        TT_FATAL((divisor_tensor.value().layout() == Layout::TILE), "divisor_tensor to nll_loss must be tilized");
        TT_FATAL(divisor_tensor.value().dtype() == DataType::BFLOAT16, "divisor tensor type must be bfloat16");
    }
}

void MorehNllLossStep2DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

void MorehNllLossStep2DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

MorehNllLossStep2DeviceOperation::spec_return_value_t MorehNllLossStep2DeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.reduction == NONE && tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.padded_shape();
    auto input_shape_without_padding = input_tensor.logical_shape();
    auto input_rank = input_shape.rank();
    auto dtype = tensor_args.input_tensor.dtype();
    Layout layout{Layout::TILE};

    ttnn::SmallVector<uint32_t> output_shape_vec;

    // Need extend 1d output to 2d, because TT not support 1d tensor
    if (input_rank == 2) {
        output_shape_vec.push_back(1);
    }

    for (uint32_t dim = 0; dim < input_rank; dim++) {
        // skip C dim
        if (dim == 1) {
            continue;
        }

        output_shape_vec.push_back(input_shape_without_padding[dim]);
    }

    auto output_shape = Shape{output_shape_vec};
    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(layout), operation_attributes.memory_config));
}

MorehNllLossStep2DeviceOperation::tensor_return_value_t MorehNllLossStep2DeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.reduction == NONE && tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }

    // In case reduction is 'sum' or 'mean' we need to create a tensor to save loss result and reduce it to
    // tensor_args.output_tensor using moreh_sum() operation
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step2

namespace ttnn::prim {
ttnn::operations::moreh::moreh_nll_loss_step2::MorehNllLossStep2DeviceOperation::tensor_return_value_t
moreh_nll_loss_step2(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::string& reduction,
    const std::optional<Tensor>& weight_tensor,
    const std::optional<Tensor>& divisor_tensor,
    const std::optional<Tensor>& output_tensor,
    int32_t ignore_index,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_nll_loss_step2::MorehNllLossStep2DeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        reduction,
        ignore_index < 0 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(ignore_index),
        memory_config.value_or(input_tensor.memory_config()),
        compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, target_tensor, weight_tensor, divisor_tensor, output_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
