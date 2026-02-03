// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>

#include <ttnn/tensor/types.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::rotate {

RotateDeviceOperation::program_factory_t RotateDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /* tensor_args */) {
    if (operation_attributes.interpolation_mode == "bilinear") {
        return BilinearProgramFactory{};
    }
    return NearestProgramFactory{};
}

void RotateDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Tensor rank
    TT_FATAL(
        input.logical_shape().rank() == 4,
        "Input tensor must be 4D (N, H, W, C), got rank {}",
        input.logical_shape().rank());

    // Layout
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input tensor must be in ROW_MAJOR layout");

    // Dtype
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Input tensor dtype must be bfloat16 or float32, got {}",
        input.dtype());

    // On device
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor must be on device");

    // Buffer allocated
    TT_FATAL(input.buffer() != nullptr, "Input tensor must be allocated in buffers on device");

    // Expand parameter
    TT_FATAL(
        !operation_attributes.expand,
        "expand=True is not supported. Only same-size rotation (expand=False) is implemented");

    // Interpolation mode validation
    TT_FATAL(
        operation_attributes.interpolation_mode == "nearest" || operation_attributes.interpolation_mode == "bilinear",
        "Only 'nearest' and 'bilinear' interpolation_mode are supported, got '{}'",
        operation_attributes.interpolation_mode);

    // Memory layout validation - width sharding is not supported
    if (input.is_sharded()) {
        auto mem_layout = input.memory_config().memory_layout();
        TT_FATAL(
            mem_layout != tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            "Width sharding is not supported for rotate operation. Got memory layout {}",
            static_cast<int>(mem_layout));
    }
}

void RotateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void RotateDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

RotateDeviceOperation::spec_return_value_t RotateDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    ttnn::Shape output_shape = input.logical_shape();
    ttnn::Shape output_padded = input.padded_shape();

    if (operation_attributes.memory_config.is_sharded()) {
        if (operation_attributes.memory_config.shard_spec().has_value()) {
            auto shard_spec = operation_attributes.memory_config.shard_spec().value();
            MemoryConfig mem_config = operation_attributes.memory_config.with_shard_spec(shard_spec);
            return TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), mem_config));
        }
        if (operation_attributes.memory_config.nd_shard_spec().has_value()) {
            return TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(
                    input.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), operation_attributes.memory_config));
        }
    }

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            input.dtype(),
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            operation_attributes.memory_config,
            output_shape,
            output_padded));
}

RotateDeviceOperation::tensor_return_value_t RotateDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t RotateDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tt::stl::hash::hash_objects_with_default_seed(
        operation_attributes.memory_config,
        operation_attributes.interpolation_mode,
        tensor_args.input.logical_shape(),
        tensor_args.input.dtype());
}

std::tuple<RotateDeviceOperation::operation_attributes_t, RotateDeviceOperation::tensor_args_t>
RotateDeviceOperation::invoke(
    const Tensor& input,
    float angle,
    const std::optional<std::tuple<float, float>>& center,
    float fill,
    bool expand,
    const std::string& interpolation_mode,
    const std::optional<MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{
            angle, center, fill, expand, interpolation_mode, memory_config.value_or(input.memory_config())},
        tensor_args_t{input}};
}

}  // namespace ttnn::operations::rotate
