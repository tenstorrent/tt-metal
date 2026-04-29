// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt_oss_swiglu_device_operation.hpp"

namespace ttnn::operations::experimental::gpt_oss_swiglu {

void GptOssSwigluDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

void GptOssSwigluDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& gate = tensor_args.gate_tensor;
    const auto& up = tensor_args.up_tensor;

    TT_FATAL(gate.storage_type() == StorageType::DEVICE, "gate must be on device");
    TT_FATAL(up.storage_type() == StorageType::DEVICE, "up must be on device");
    TT_FATAL(gate.device() == up.device(), "gate and up must be on the same device");

    TT_FATAL(gate.layout() == Layout::TILE, "gate must be TILE layout");
    TT_FATAL(up.layout() == Layout::TILE, "up must be TILE layout");

    TT_FATAL(gate.dtype() == DataType::BFLOAT16, "gate must be BFLOAT16, got {}", gate.dtype());
    TT_FATAL(up.dtype() == DataType::BFLOAT16, "up must be BFLOAT16, got {}", up.dtype());

    TT_FATAL(
        gate.padded_shape() == up.padded_shape(),
        "gate and up must have identical padded shape, got {} vs {}",
        gate.padded_shape(),
        up.padded_shape());

    // Sharding requirements: both inputs must be block-sharded with identical
    // shard spec so a single compute kernel can run on the same core grid for
    // both. Output inherits the same layout from output_memory_config.
    TT_FATAL(gate.is_sharded(), "gate must be sharded (BLOCK_SHARDED L1)");
    TT_FATAL(up.is_sharded(), "up must be sharded (BLOCK_SHARDED L1)");

    auto gate_mem = gate.memory_config();
    auto up_mem = up.memory_config();
    TT_FATAL(
        gate_mem.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "gate must be BLOCK_SHARDED, got {}",
        gate_mem.memory_layout());
    TT_FATAL(
        up_mem.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "up must be BLOCK_SHARDED, got {}",
        up_mem.memory_layout());
    TT_FATAL(gate.shard_spec().value() == up.shard_spec().value(), "gate and up must have identical shard specs");

    TT_FATAL(attrs.clamp_limit > 0.0f, "clamp_limit must be positive, got {}", attrs.clamp_limit);
}

GptOssSwigluDeviceOperation::spec_return_value_t GptOssSwigluDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& gate = tensor_args.gate_tensor;
    return TensorSpec(
        gate.logical_shape(),
        tt::tt_metal::TensorLayout(gate.dtype(), tt::tt_metal::PageConfig(Layout::TILE), attrs.output_memory_config));
}

GptOssSwigluDeviceOperation::tensor_return_value_t GptOssSwigluDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.gate_tensor.device());
}

std::tuple<GptOssSwigluDeviceOperation::operation_attributes_t, GptOssSwigluDeviceOperation::tensor_args_t>
GptOssSwigluDeviceOperation::invoke(
    const Tensor& gate_tensor,
    const Tensor& up_tensor,
    float alpha,
    float clamp_limit,
    const std::optional<MemoryConfig>& output_memory_config) {
    return {
        operation_attributes_t{
            .alpha = alpha,
            .clamp_limit = clamp_limit,
            .output_memory_config = output_memory_config.value_or(gate_tensor.memory_config()),
        },
        tensor_args_t{.gate_tensor = gate_tensor, .up_tensor = up_tensor},
    };
}

}  // namespace ttnn::operations::experimental::gpt_oss_swiglu

namespace ttnn::experimental {

Tensor gpt_oss_swiglu(
    const ttnn::Tensor& gate_tensor,
    const ttnn::Tensor& up_tensor,
    float alpha,
    float clamp_limit,
    const std::optional<ttnn::MemoryConfig>& output_memory_config) {
    auto [attrs, args] = operations::experimental::gpt_oss_swiglu::GptOssSwigluDeviceOperation::invoke(
        gate_tensor, up_tensor, alpha, clamp_limit, output_memory_config);
    return ttnn::device_operation::launch<operations::experimental::gpt_oss_swiglu::GptOssSwigluDeviceOperation>(
        attrs, args);
}

}  // namespace ttnn::experimental
