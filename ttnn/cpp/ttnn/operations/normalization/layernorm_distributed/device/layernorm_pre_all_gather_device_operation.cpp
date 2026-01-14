// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::normalization {

LayerNormPreAllGatherDeviceOperation::program_factory_t LayerNormPreAllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    // Check if 2D core grid is requested
    if (args.use_2d_core_grid.has_value() && args.use_2d_core_grid.value()) {
        return program::LayerNormPreAllGather2DProgramFactory{};
    }

    // Check if Welford algorithm is requested (only for layernorm)
    if (std::holds_alternative<LayerNormDefaultProgramConfig>(args.program_config)) {
        const auto& program_config = std::get<LayerNormDefaultProgramConfig>(args.program_config);
        if (program_config.use_welford) {
            return program::LayerNormPreAllGatherWelfordProgramFactory{};
        }
    }

    // Default to normal program factory
    return program::LayerNormPreAllGatherProgramFactory{};
}

void LayerNormPreAllGatherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void LayerNormPreAllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& tensor = tensor_args.input;

    TT_FATAL(tensor.layout() == Layout::TILE, "Only tilized inputs supported.");
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved inputs supported.");
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B ||
            tensor.dtype() == DataType::FLOAT32,
        "Input data format not supported.");
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    // Additional validation for Welford - it doesn't support rmsnorm
    if (std::holds_alternative<LayerNormDefaultProgramConfig>(args.program_config)) {
        const auto& program_config = std::get<LayerNormDefaultProgramConfig>(args.program_config);
        if (program_config.use_welford && args.norm_type == LayerNormDistributedType::RMSNORM) {
            TT_FATAL(false, "RMS norm is not compatible with Welford algorithm. Please disable use_welford flag.");
        }
    }

    // Additional validation for 2D core grid - it doesn't support Welford
    if (args.use_2d_core_grid.has_value() && args.use_2d_core_grid.value()) {
        if (std::holds_alternative<LayerNormDefaultProgramConfig>(args.program_config)) {
            const auto& program_config = std::get<LayerNormDefaultProgramConfig>(args.program_config);
            if (program_config.use_welford) {
                TT_FATAL(false, "Welford layernorm variation does not support 2D core grid.");
            }
        }
    }
}

LayerNormPreAllGatherDeviceOperation::spec_return_value_t LayerNormPreAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    auto output_shape = input_tensor.logical_shape();
    uint32_t num_tiles_w = 1;
    if (args.norm_type == LayerNormDistributedType::LAYERNORM) {
        num_tiles_w = 2;
    }
    output_shape[3] = num_tiles_w * TILE_WIDTH;

    auto output_dtype = args.dtype.value_or(input_tensor.dtype());
    return TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::TILE), input_tensor.memory_config()));
}

LayerNormPreAllGatherDeviceOperation::tensor_return_value_t LayerNormPreAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::normalization

namespace ttnn::prim {

Tensor layer_norm_pre_all_gather(
    const Tensor& input,
    ttnn::operations::normalization::LayerNormDistributedType norm_type,
    const std::optional<tt::tt_metal::DataType>& dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const ttnn::operations::normalization::LayerNormProgramConfig& program_config,
    const std::optional<bool>& use_2d_core_grid) {
    using OperationType = ttnn::operations::normalization::LayerNormPreAllGatherDeviceOperation;
    return ttnn::device_operation::detail::launch<OperationType>(
        OperationType::operation_attributes_t{
            .norm_type = norm_type,
            .dtype = dtype,
            .compute_kernel_config = compute_kernel_config,
            .program_config = program_config,
            .use_2d_core_grid = use_2d_core_grid,
        },
        OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
