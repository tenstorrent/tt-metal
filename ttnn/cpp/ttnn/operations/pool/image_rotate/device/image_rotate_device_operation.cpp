// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "image_rotate_device_operation.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::image_rotate {
using namespace tt;
using namespace tt::tt_metal;

ImageRotateDeviceOperation::program_factory_t ImageRotateDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void ImageRotateDeviceOperation::validate_inputs(
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

    // Interleaved memory
    TT_FATAL(
        !input.memory_config().is_sharded(), "Input tensor must be DRAM interleaved, sharded memory not supported");

    // Channel alignment
    const uint32_t C = input.logical_shape()[-1];
    const uint32_t element_size = input.element_size();
    TT_FATAL(
        (C * element_size) % 32 == 0,
        "Channel dimension must be aligned to 32 bytes, got {} channels * {} bytes = {} bytes",
        C,
        element_size,
        C * element_size);

    // Expand parameter
    TT_FATAL(
        !operation_attributes.expand,
        "expand=True is not supported. Only same-size rotation (expand=False) is implemented");
}

void ImageRotateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void ImageRotateDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

ImageRotateDeviceOperation::spec_return_value_t ImageRotateDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Output shape is the same as input shape for image_rotate (expand=False)
    ttnn::Shape output_shape = input.logical_shape();
    ttnn::Shape output_padded = input.padded_shape();

    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            input.dtype(),
            PageConfig(Layout::ROW_MAJOR),
            operation_attributes.memory_config,
            output_shape,
            output_padded));
}

ImageRotateDeviceOperation::tensor_return_value_t ImageRotateDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t ImageRotateDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Cache based on tensor shape and memory config only
    // angle, center, and fill are runtime args and don't affect program structure
    // expand is validated to be false, so doesn't need to be in hash
    return tt::stl::hash::hash_objects_with_default_seed(
        operation_attributes.memory_config, tensor_args.input.logical_shape(), tensor_args.input.dtype());
}

std::tuple<ImageRotateDeviceOperation::operation_attributes_t, ImageRotateDeviceOperation::tensor_args_t>
ImageRotateDeviceOperation::invoke(
    const Tensor& input,
    float angle,
    const std::optional<std::tuple<float, float>>& center,
    float fill,
    bool expand,
    const std::optional<MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{angle, center, fill, expand, memory_config.value_or(input.memory_config())},
        tensor_args_t{input}};
}

}  // namespace ttnn::operations::image_rotate
