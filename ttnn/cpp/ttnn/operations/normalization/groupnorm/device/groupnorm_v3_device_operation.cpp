// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_v3_device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::group_norm_v3 {

GroupNormV3DeviceOperation::program_factory_t GroupNormV3DeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return program::GroupNormV3ProgramFactory{};
}

void GroupNormV3DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void GroupNormV3DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& device = input.device();

    // TODO: Chunk size validation
    // Must be divisible by TILE_HW
    // Must be less than or equal to (C * H * W / num_groups)

    // Check that the number of cores needed are available
    const auto& num_available_cores =
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
    const auto& num_required_cores = args.core_grid.x * args.core_grid.y;
    TT_FATAL(
        num_required_cores <= num_available_cores,
        "Requested core_grid {} has {} cores, but device has a maximum grid of {} with {} cores",
        args.core_grid,
        num_required_cores,
        device->compute_with_storage_grid_size(),
        num_available_cores);
}

spec_return_value_t GroupNormV3DeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    if (args.inplace) {
        return input.tensor_spec();
    }

    return TensorSpec(
        input.logical_shape(), TensorLayout(args.output_dtype, PageConfig(input.layout()), args.output_mem_config));
}

tensor_return_value_t GroupNormV3DeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    if (args.inplace) {
        return input;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), input.device());
}

}  // namespace ttnn::operations::normalization::group_norm_v3

namespace ttnn::prim {

ttnn::operations::normalization::group_norm_v3::tensor_return_value_t group_norm_v3(
    const Tensor& input,
    uint32_t num_groups,
    float eps,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const CoreCoord& core_grid,
    bool inplace,
    int chunk_size,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<Tensor> gamma,
    std::optional<Tensor> beta) {
    using OperationType = ttnn::operations::normalization::group_norm_v3::GroupNormV3DeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_groups = num_groups,
        .eps = eps,
        .output_dtype = output_dtype,
        .output_mem_config = output_mem_config,
        .core_grid = core_grid,
        .inplace = inplace,
        .chunk_size = chunk_size,
        .compute_kernel_config = compute_kernel_config,
    };

    auto tensor_args = OperationType::tensor_args_t{
        .input = input,
        .gamma = std::move(gamma),
        .beta = std::move(beta),
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
