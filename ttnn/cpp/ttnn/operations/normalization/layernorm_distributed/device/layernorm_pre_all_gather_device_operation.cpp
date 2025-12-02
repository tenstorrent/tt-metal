// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>

#include <optional>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization::layernorm {

LayerNormPreAllGatherDeviceOperation::program_factory_t LayerNormPreAllGatherDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Use 2D factory if use_2d_core_grid is set to true
    if (args.use_2d_core_grid.has_value() && args.use_2d_core_grid.value()) {
        return program::LayerNormPreAllGather2DProgramFactory{};
    } else {
        return program::LayerNormPreAllGatherProgramFactory{};
    }
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
}

LayerNormPreAllGatherDeviceOperation::spec_return_value_t LayerNormPreAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    auto output_shape = input_tensor.logical_shape();
    const uint32_t num_tiles_w = (args.norm_type == LayerNormDistributedType::LAYERNORM) ? 2 : 1;
    output_shape[3] = num_tiles_w * TILE_WIDTH;

    return TensorSpec(output_shape, TensorLayout(args.dtype, PageConfig(Layout::TILE), input_tensor.memory_config()));
}

LayerNormPreAllGatherDeviceOperation::tensor_return_value_t LayerNormPreAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

std::tuple<
    LayerNormPreAllGatherDeviceOperation::operation_attributes_t,
    LayerNormPreAllGatherDeviceOperation::tensor_args_t>
LayerNormPreAllGatherDeviceOperation::invoke(
    const Tensor& input,
    LayerNormDistributedType norm_type,
    DataType dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<bool> use_2d_core_grid,
    const LayerNormDistributedDefaultProgramConfig& program_config,
    const std::optional<Tensor>& preallocated_output) {
    return {
        operation_attributes_t{
            .norm_type = norm_type,
            .dtype = dtype,
            .compute_kernel_config = compute_kernel_config,
            .use_2d_core_grid = use_2d_core_grid,
            .program_config = program_config},
        tensor_args_t{.input = input, .preallocated_output = preallocated_output}};
}

}  // namespace ttnn::operations::normalization::layernorm
