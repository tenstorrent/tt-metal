// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

BatchNormDeviceOperation::program_factory_t BatchNormDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return BatchNormProgramFactory{};
}

void BatchNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to batch_norm need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands to batch_norm need to be allocated in buffers on device!");
    TT_FATAL(input.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", input.layout());
    TT_FATAL(!input.is_sharded(), "Quasar batch_norm does not support sharded input.");
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(), "Quasar batch_norm does not support sharded output.");
    TT_FATAL(input.logical_shape().rank() == 4, "batch_norm supports tensors of rank 4");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16,
        "Quasar batch_norm: input must be BFLOAT16 (no BFLOAT8_B on Quasar; FLOAT32 not implemented), got: {}",
        input.dtype());

    const uint32_t channels = input.logical_shape()[1];
    auto check_param = [&](const Tensor& t, const char* name) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
        TT_FATAL(t.buffer() != nullptr, "{} must be allocated in a buffer on device", name);
        TT_FATAL(t.device() == input.device(), "{} must be on the same device as the input", name);
        TT_FATAL(t.layout() == Layout::TILE, "{} must have TILE layout, got: {}", name, t.layout());
        TT_FATAL(t.dtype() == DataType::BFLOAT16, "Quasar batch_norm: {} must be BFLOAT16, got: {}", name, t.dtype());
        TT_FATAL(!t.is_sharded(), "{} must be interleaved", name);
        TT_FATAL(t.logical_shape().rank() == 4, "{} must have rank 4", name);
        TT_FATAL(
            t.logical_shape()[0] == 1 && t.logical_shape()[2] == 1 && t.logical_shape()[3] == 1,
            "{} must be [1, C, 1, 1], got {}",
            name,
            t.logical_shape());
        TT_FATAL(
            t.logical_shape()[1] == channels,
            "{}[1] ({}) must match the input's channel count ({})",
            name,
            t.logical_shape()[1],
            channels);
    };
    check_param(tensor_args.running_mean, "running_mean");
    check_param(tensor_args.running_var, "running_var");
    if (tensor_args.weight.has_value()) {
        check_param(*tensor_args.weight, "weight");
    }
    if (tensor_args.bias.has_value()) {
        check_param(*tensor_args.bias, "bias");
    }
}

TensorSpec BatchNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(input.dtype(), PageConfig(input.layout()), operation_attributes.output_mem_config));
}

Tensor BatchNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor batch_norm(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    float eps,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    return ttnn::device_operation::launch<BatchNormDeviceOperation>(
        BatchNormParams{
            .eps = eps,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
        },
        BatchNormInputs{
            .input = input,
            .running_mean = running_mean,
            .running_var = running_var,
            .weight = weight,
            .bias = bias,
        });
}

}  // namespace ttnn::prim::qsr
