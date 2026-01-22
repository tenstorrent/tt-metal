// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm_w_rm_device_operation.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::normalization::layer_norm_w_rm {
using namespace tt;
using namespace tt::tt_metal;

LayerNormWRmDeviceOperation::program_factory_t LayerNormWRmDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::LayerNormWRmProgramFactory{};
}

void LayerNormWRmDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void LayerNormWRmDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Extract parameters from operation_attributes for use in validations
    const auto epsilon = operation_attributes.epsilon;

    // Extract tensors from tensor_args
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    // Suppress unused variable warnings if parameters aren't used in validations
    (void)epsilon;

    // Storage type validation
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input_tensor tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "input_tensor tensor must be allocated");
    TT_FATAL(gamma.storage_type() == StorageType::DEVICE, "gamma tensor must be on device");
    TT_FATAL(gamma.buffer() != nullptr, "gamma tensor must be allocated");
    TT_FATAL(beta.storage_type() == StorageType::DEVICE, "beta tensor must be on device");
    TT_FATAL(beta.buffer() != nullptr, "beta tensor must be allocated");

    // Validations from spec
    TT_FATAL(
        input.logical_shape().rank() >= 2,
        "Input tensor must have at least 2 dimensions, got rank {}",
        input.logical_shape().rank());
    TT_FATAL(
        input.layout() == Layout::ROW_MAJOR,
        "Input tensor must be in ROW_MAJOR layout");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must be interleaved");
    TT_FATAL(
        input.is_allocated(),
        "Input tensor must be on device");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Unsupported dtype: only BFLOAT16 and FLOAT32 are supported, got {}",
        input.dtype());
    TT_FATAL(
        gamma.logical_shape()[-1] == input.logical_shape()[-1],
        "Gamma shape must match input width, expected {} got {}",
        input.logical_shape()[-1],        gamma.logical_shape()[-1]);
    TT_FATAL(
        gamma.layout() == Layout::ROW_MAJOR,
        "Gamma must be in ROW_MAJOR layout");
    TT_FATAL(
        gamma.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Gamma must be interleaved in DRAM");
    TT_FATAL(
        gamma.device() == input.device(),
        "Gamma must be on same device as input");
    TT_FATAL(
        gamma.dtype() == input.dtype(),
        "Gamma dtype must match input dtype, expected {} got {}",
        input.dtype(),        gamma.dtype());
    TT_FATAL(
        beta.logical_shape()[-1] == input.logical_shape()[-1],
        "Beta shape must match input width, expected {} got {}",
        input.logical_shape()[-1],        beta.logical_shape()[-1]);
    TT_FATAL(
        beta.layout() == Layout::ROW_MAJOR,
        "Beta must be in ROW_MAJOR layout");
    TT_FATAL(
        beta.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Beta must be interleaved in DRAM");
    TT_FATAL(
        beta.device() == input.device(),
        "Beta must be on same device as input");
    TT_FATAL(
        beta.dtype() == input.dtype(),
        "Beta dtype must match input dtype, expected {} got {}",
        input.dtype(),        beta.dtype());
    TT_FATAL(
        epsilon > 0.0f,
        "Epsilon must be positive, got {}",
        epsilon);
}

LayerNormWRmDeviceOperation::spec_return_value_t LayerNormWRmDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Output shape computation
    ttnn::Shape output_shape = input.logical_shape();
    ttnn::Shape output_padded = input.padded_shape();

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

tt::stl::hash::hash_t LayerNormWRmDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<LayerNormWRmDeviceOperation>(
        operation_attributes,
        input.dtype(),
        input.memory_config(),
        input_shape);

    return hash;
}

LayerNormWRmDeviceOperation::tensor_return_value_t LayerNormWRmDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::operations::normalization::layer_norm_w_rm

namespace ttnn::prim {
ttnn::operations::normalization::layer_norm_w_rm::LayerNormWRmDeviceOperation::tensor_return_value_t layer_norm_w_rm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::normalization::layer_norm_w_rm::LayerNormWRmDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .epsilon = epsilon,
            .output_mem_config = output_mem_config
        },
        OperationType::tensor_args_t{
            .input = input,            .gamma = gamma,            .beta = beta        });
}
}  // namespace ttnn::prim