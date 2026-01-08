// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_avg_w_rm_device_operation.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::reduction::reduce_avg_w_rm {
using namespace tt;
using namespace tt::tt_metal;

ReduceAvgWRmDeviceOperation::program_factory_t ReduceAvgWRmDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::ReduceAvgWRmProgramFactory{};
}

void ReduceAvgWRmDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ReduceAvgWRmDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Storage type validation
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input_tensor tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "input_tensor tensor must be allocated");

    // Validations from spec
    TT_FATAL(
        input.logical_shape().rank() == 4, "Input tensor must have rank 4, got rank {}", input.logical_shape().rank());
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input tensor must be in ROW_MAJOR layout");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must have INTERLEAVED memory layout");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16");
    TT_FATAL(input.logical_shape()[-1] > 0, "Input width must be positive");
    TT_FATAL(
        input.logical_shape()[-2] % 32 == 0,
        "Input height must be multiple of TILE_HEIGHT (32), got {}",
        input.logical_shape()[-2]);
    TT_FATAL(
        input.logical_shape()[-1] % 32 == 0,
        "Input width must be multiple of TILE_WIDTH (32), got {}",
        input.logical_shape()[-1]);
    TT_FATAL(input.is_allocated(), "Input must be allocated on device");
}

spec_return_value_t ReduceAvgWRmDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Output shape computation
    ttnn::SmallVector<uint32_t> dims(input.logical_shape().cbegin(), input.logical_shape().cend());
    dims.back() = 32;
    ttnn::Shape output_shape(dims);
    ttnn::SmallVector<uint32_t> pdims(input.logical_shape().cbegin(), input.logical_shape().cend());
    pdims.back() = 32;
    ttnn::Shape output_padded(pdims);

    auto output_dtype = input.dtype();
    auto output_mem_config = operation_attributes.output_mem_config.value_or(input.memory_config());

    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            output_dtype, PageConfig(Layout::ROW_MAJOR), output_mem_config, output_shape, output_padded));
}

tt::stl::hash::hash_t ReduceAvgWRmDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<ReduceAvgWRmDeviceOperation>(
        operation_attributes, input.dtype(), input.memory_config(), input_shape);

    return hash;
}

tensor_return_value_t ReduceAvgWRmDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::operations::reduction::reduce_avg_w_rm

namespace ttnn::prim {
ttnn::operations::reduction::reduce_avg_w_rm::ReduceAvgWRmDeviceOperation::tensor_return_value_t reduce_avg_w_rm(
    const Tensor& input,
    std::optional<MemoryConfig> output_mem_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::operations::reduction::reduce_avg_w_rm::ReduceAvgWRmDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = output_mem_config, .compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
