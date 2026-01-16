// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fused_rm_device_operation.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::normalization::layernorm_fused_rm {
using namespace tt;
using namespace tt::tt_metal;

LayernormFusedRmDeviceOperation::program_factory_t LayernormFusedRmDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::LayernormFusedRmProgramFactory{};
}

void LayernormFusedRmDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void LayernormFusedRmDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto epsilon = operation_attributes.epsilon;

    // Storage type validation
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input_tensor tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "input_tensor tensor must be allocated");
    TT_FATAL(gamma.storage_type() == StorageType::DEVICE, "gamma tensor must be on device");
    TT_FATAL(gamma.buffer() != nullptr, "gamma tensor must be allocated");
    TT_FATAL(beta.storage_type() == StorageType::DEVICE, "beta tensor must be on device");
    TT_FATAL(beta.buffer() != nullptr, "beta tensor must be allocated");

    // Validations from spec
    TT_FATAL(input.is_allocated(), "Input must be allocated on device");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input must be in ROW_MAJOR layout");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input must be in INTERLEAVED memory");
    TT_FATAL(
        input.logical_shape().rank() >= 2,
        "Input must have at least 2 dimensions, got rank {}",
        input.logical_shape().rank());
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input must be BFLOAT16, got {}", input.dtype());
    TT_FATAL(input.logical_shape()[-1] % 32 == 0, "Width must be multiple of 32, got {}", input.logical_shape()[-1]);
    TT_FATAL(input.logical_shape()[-2] % 32 == 0, "Height must be multiple of 32, got {}", input.logical_shape()[-2]);
    TT_FATAL(gamma.is_allocated(), "Gamma must be allocated on device");
    TT_FATAL(gamma.layout() == Layout::ROW_MAJOR, "Gamma must be in ROW_MAJOR layout");
    TT_FATAL(
        gamma.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Gamma must be in INTERLEAVED memory");
    TT_FATAL(gamma.dtype() == DataType::BFLOAT16, "Gamma must be BFLOAT16, got {}", gamma.dtype());
    TT_FATAL(
        gamma.logical_shape()[-1] == input.logical_shape()[-1],
        "Gamma last dimension {} must match input last dimension {}",
        gamma.logical_shape()[-1],
        input.logical_shape()[-1]);
    TT_FATAL(beta.is_allocated(), "Beta must be allocated on device");
    TT_FATAL(beta.layout() == Layout::ROW_MAJOR, "Beta must be in ROW_MAJOR layout");
    TT_FATAL(
        beta.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Beta must be in INTERLEAVED memory");
    TT_FATAL(beta.dtype() == DataType::BFLOAT16, "Beta must be BFLOAT16, got {}", beta.dtype());
    TT_FATAL(
        beta.logical_shape()[-1] == input.logical_shape()[-1],
        "Beta last dimension {} must match input last dimension {}",
        beta.logical_shape()[-1],
        input.logical_shape()[-1]);
    TT_FATAL(epsilon > 0.0f, "Epsilon must be positive, got {}", epsilon);
}

LayernormFusedRmDeviceOperation::spec_return_value_t LayernormFusedRmDeviceOperation::compute_output_specs(
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

tt::stl::hash::hash_t LayernormFusedRmDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<LayernormFusedRmDeviceOperation>(
        operation_attributes, input.dtype(), input.memory_config(), input_shape);

    return hash;
}

LayernormFusedRmDeviceOperation::tensor_return_value_t LayernormFusedRmDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::operations::normalization::layernorm_fused_rm

namespace ttnn::prim {
ttnn::operations::normalization::layernorm_fused_rm::LayernormFusedRmDeviceOperation::tensor_return_value_t
layernorm_fused_rm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::normalization::layernorm_fused_rm::LayernormFusedRmDeviceOperation;
    return ttnn::device_operation::detail::launch<OperationType>(
        OperationType::operation_attributes_t{.epsilon = epsilon, .output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input, .gamma = gamma, .beta = beta});
}
}  // namespace ttnn::prim
