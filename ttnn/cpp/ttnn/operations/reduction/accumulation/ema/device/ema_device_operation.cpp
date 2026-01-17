// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "ema_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt_stl/assert.hpp>

#include <cmath>

namespace ttnn::prim {

using namespace tt::tt_metal;

EmaDeviceOperation::program_factory_t EmaDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return EmaProgramFactory{};
}

void EmaDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void EmaDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    // Dtype, Device and layout checks
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16, got: {}", input_tensor.dtype());
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Input tensor must be on device, got: {}",
        input_tensor.storage_type());
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in a device buffer");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", input_tensor.layout());

    // Shape constraints: [1, B, C, T]
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "EMA input must be 4D [1, B, C, T], got rank {}", input_shape.rank());
    TT_FATAL(input_shape[0] == 1, "EMA expects leading dimension to be 1, got {}", input_shape[0]);

    // This OP produces as many elements in output as there are in input
    // Thus, the volume must be the same to avoid writing outside the output buffer
    if (tensor_args.optional_output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.optional_output_tensor.value();
        TT_FATAL(
            output_tensor.dtype() == DataType::BFLOAT16,
            "Output tensor must be BFLOAT16, got: {}",
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Output tensor must be on device, got: {}",
            output_tensor.storage_type());
        TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor must be allocated in a device buffer");
        TT_FATAL(
            output_tensor.layout() == Layout::TILE,
            "Output tensor must have TILE layout, got: {}",
            output_tensor.layout());
        TT_FATAL(
            input_tensor.padded_shape().volume() == output_tensor.padded_shape().volume(),
            "Input and output must have the same volume, input: {}, output: {}",
            input_tensor.padded_shape().volume(),
            output_tensor.padded_shape().volume());
    }

    // Alpha validation
    TT_FATAL(!std::isnan(operation_attributes.alpha), "EMA alpha must be a valid number, got NaN");
}

TensorSpec EmaDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    return tensor_args.input.tensor_spec().with_memory_config(operation_attributes.output_mem_config);
}

Tensor EmaDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

ttnn::Tensor ema_device(
    const Tensor& input,
    float alpha,
    CoreCoord grid_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = EmaDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .alpha = alpha,
            .grid_size = grid_size,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
        },
        OperationType::tensor_args_t{
            .input = input,
            .optional_output_tensor = std::move(optional_output_tensor),
        });
}

}  // namespace ttnn::prim
