// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "dispatch_tilize_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize {

void DispatchTilizeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input = tensor_args.input_tensor;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "dispatch_tilize input must be on device");
    TT_FATAL(input.buffer() != nullptr, "dispatch_tilize input must have a buffer");

    const auto& counts = tensor_args.total_counts_per_expert;
    if (counts.has_value()) {
        TT_FATAL(
            operation_attributes.experts_per_chip > 0, "experts_per_chip must be > 0 for the region-aware (skip) path");
        TT_FATAL(counts->storage_type() == StorageType::DEVICE, "total_counts_per_expert must be on device");
        const uint32_t num_experts = counts->logical_shape()[-1];
        TT_FATAL(
            num_experts % operation_attributes.experts_per_chip == 0,
            "total_counts_per_expert width ({}) must be divisible by experts_per_chip ({})",
            num_experts,
            operation_attributes.experts_per_chip);
    }
}

void DispatchTilizeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const ttnn::Tensor& input = tensor_args.input_tensor;
    TT_FATAL(input.layout() == ttnn::Layout::ROW_MAJOR, "dispatch_tilize input must be ROW_MAJOR");
    // The writer only implements the interleaved path (no sharded branch), so both sides must be interleaved.
    TT_FATAL(!input.memory_config().is_sharded(), "dispatch_tilize input must be interleaved");
    TT_FATAL(!operation_attributes.output_memory_config.is_sharded(), "dispatch_tilize output must be interleaved");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FP8_E4M3,
        "dispatch_tilize input must be bfloat16 or fp8_e4m3, got {}",
        input.dtype());
    TT_FATAL(
        operation_attributes.output_dtype == DataType::BFLOAT8_B ||
            operation_attributes.output_dtype == DataType::BFLOAT16,
        "dispatch_tilize output dtype must be bfloat8_b or bfloat16, got {}",
        operation_attributes.output_dtype);

    if (tensor_args.total_counts_per_expert.has_value()) {
        const auto& counts = *tensor_args.total_counts_per_expert;
        TT_FATAL(counts.layout() == ttnn::Layout::ROW_MAJOR, "total_counts_per_expert must be ROW_MAJOR");
        TT_FATAL(counts.dtype() == DataType::UINT32, "total_counts_per_expert must be uint32");
    }
}

ttnn::TensorSpec DispatchTilizeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input = tensor_args.input_tensor;
    return TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(Layout::TILE),
            operation_attributes.output_memory_config));
}

ttnn::Tensor DispatchTilizeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize

namespace ttnn::prim {

ttnn::Tensor dispatch_tilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& total_counts_per_expert,
    tt::tt_metal::DataType output_dtype,
    uint32_t experts_per_chip,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    namespace dt = ttnn::operations::experimental::deepseek_prefill::dispatch_tilize;
    using OperationType = dt::DispatchTilizeDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        dt::DispatchTilizeParams{output_dtype, output_memory_config, experts_per_chip},
        dt::DispatchTilizeInputs{input_tensor, total_counts_per_expert});
}

}  // namespace ttnn::prim
