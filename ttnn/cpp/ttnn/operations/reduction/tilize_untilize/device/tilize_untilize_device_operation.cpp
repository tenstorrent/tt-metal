// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_untilize_device_operation.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::reduction {
using namespace tt;
using namespace tt::tt_metal;

TilizeUntilizeDeviceOperation::program_factory_t TilizeUntilizeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::TilizeUntilizeProgramFactory{};
}

void TilizeUntilizeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void TilizeUntilizeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Storage type validation
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input_tensor tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "input_tensor tensor must be allocated");

    // Validations from spec
    TT_FATAL(
        input.logical_shape().rank() == 4, "Input tensor must be 4D (NCHW), got rank {}", input.logical_shape().rank());
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input must be in ROW_MAJOR layout, got {}", input.layout());
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input must be interleaved (not sharded)");
    TT_FATAL(input.is_allocated(), "Input must be allocated on device");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Unsupported dtype {}, expected BFLOAT16 or FLOAT32",
        input.dtype());
    TT_FATAL(
        input.logical_shape()[-2] % 32 == 0,
        "Height must be multiple of TILE_HEIGHT (32), got {}",
        input.logical_shape()[-2]);
    TT_FATAL(
        input.logical_shape()[-1] % 32 == 0,
        "Width must be multiple of TILE_WIDTH (32), got {}",
        input.logical_shape()[-1]);
}

spec_return_value_t TilizeUntilizeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Output shape computation
    auto output_shape = input.logical_shape();
    auto output_padded = input.logical_shape();

    auto output_dtype = input.dtype();

    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            operation_attributes.output_mem_config,
            output_shape,
            output_padded));
}

tt::stl::hash::hash_t TilizeUntilizeDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<TilizeUntilizeDeviceOperation>(
        operation_attributes, input.dtype(), input.memory_config(), input_shape);

    return hash;
}

tensor_return_value_t TilizeUntilizeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

std::tuple<TilizeUntilizeDeviceOperation::operation_attributes_t, TilizeUntilizeDeviceOperation::tensor_args_t>
TilizeUntilizeDeviceOperation::invoke(
    const Tensor& input,
    std::optional<MemoryConfig> output_memory_config,
    std::optional<DataType> output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{
            .output_memory_config = output_memory_config,
            .output_dtype = output_dtype,
            .output_mem_config = memory_config.value_or(input.memory_config())},
        tensor_args_t{.input = input}};
}

}  // namespace ttnn::operations::reduction
