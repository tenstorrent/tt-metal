// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_device_operation.hpp"
#include "tilize_device_operation_types.hpp"
#include "tilize_multi_core_interleaved_program_factory.hpp"
#include "tilize_multi_core_block_program_factory.hpp"
#include "tilize_single_core_program_factory.hpp"
#include "tilize_multi_core_sharded_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void TilizeDeviceOperation::validate_on_program_cache_hit(
    const TilizeDeviceOperation::operation_attributes_t& args,
    const TilizeDeviceOperation::tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void TilizeDeviceOperation::validate_on_program_cache_miss(
    const TilizeDeviceOperation::operation_attributes_t& operation_attributes,
    const TilizeDeviceOperation::tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to tilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");

    TT_FATAL(
        input_tensor_a.physical_volume() % tt::constants::TILE_HW == 0,
        "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
        input_tensor_a.physical_volume(),
        tt::constants::TILE_HW);

    auto width = input_tensor_a.padded_shape()[-1];
    uint32_t stick_s = width;
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::FLOAT32 or
            input_tensor_a.dtype() == DataType::UINT32 or input_tensor_a.dtype() == DataType::INT32 or
            input_tensor_a.dtype() == DataType::UINT16,
        "data type must be bfloat16, float32, uint32, int32, or uint16");

    uint32_t stick_size = stick_s * input_tensor_a.element_size();  // Assuming bfloat16 dataformat

    TT_FATAL((stick_size % 2) == 0, "Stick size must be divisible by 2");

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Input tensor memory layout must be HEIGHT_SHARDED but got {}",
            input_tensor_a.memory_config().memory_layout());
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(),
            "Output memory config layout ({}) must match input tensor memory layout ({})",
            operation_attributes.output_mem_config.memory_layout(),
            input_tensor_a.memory_config().memory_layout());
        TT_FATAL(operation_attributes.use_multicore == true, "Multicore must be enabled for sharded input");
        TT_FATAL(
            input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Input tensor shard orientation must be ROW_MAJOR but got {}",
            input_tensor_a.shard_spec().value().orientation);
    } else {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input tensor memory layout must be INTERLEAVED but got {}",
            input_tensor_a.memory_config().memory_layout());
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Output memory config layout must be INTERLEAVED but got {}",
            operation_attributes.output_mem_config.memory_layout());
    }
}

TilizeDeviceOperation::spec_return_value_t TilizeDeviceOperation::compute_output_specs(
    const TilizeDeviceOperation::operation_attributes_t& operation_attributes,
    const TilizeDeviceOperation::tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    if (input_tensor.memory_config().is_sharded()) {
        auto mem_config =
            operation_attributes.output_mem_config.with_shard_spec(input_tensor.memory_config().shard_spec());
        return {TensorSpec(
            input_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                operation_attributes.output_dtype,
                PageConfig(Layout::TILE),
                mem_config,
                input_tensor.logical_shape(),
                input_tensor.padded_shape()))};
    }

    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            operation_attributes.output_dtype,
            PageConfig(Layout::TILE),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

TilizeDeviceOperation::program_factory_t TilizeDeviceOperation::select_program_factory(
    const TilizeDeviceOperation::operation_attributes_t& operation_attributes,
    const TilizeDeviceOperation::tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;

    bool use_single_core = (operation_attributes.use_low_perf) || (!operation_attributes.use_multicore) ||
                           (operation_attributes.sub_core_grids.has_value() &&
                            (operation_attributes.sub_core_grids.value().num_cores() < 2));
    if (use_single_core) {
        return program::TilizeSingleCoreProgramFactory{};
    }

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(
            !operation_attributes.sub_core_grids.has_value(),
            "Sharded tilize does not support sub core grid specification");
        return program::TilizeMultiCoreShardedProgramFactory{};
    }
    if (!operation_attributes.enough_space_height) {
        return program::TilizeMultiCoreBlockProgramFactory{};
    }

    return program::TilizeMultiCoreInterleavedProgramFactory{};
}

TilizeDeviceOperation::tensor_return_value_t TilizeDeviceOperation::create_output_tensors(
    const TilizeDeviceOperation::operation_attributes_t& args,
    const TilizeDeviceOperation::tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<TilizeDeviceOperation::operation_attributes_t, TilizeDeviceOperation::tensor_args_t>
TilizeDeviceOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DataType>& output_dtype,
    bool use_multicore,
    bool enough_space_width,
    bool enough_space_height,
    bool use_low_perf,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return {
        TilizeDeviceOperation::operation_attributes_t{
            .output_mem_config = output_mem_config.value_or(input_tensor.memory_config()),
            .output_dtype = output_dtype.value_or(input_tensor.dtype()),
            .use_multicore = use_multicore,
            .enough_space_width = enough_space_width,
            .enough_space_height = enough_space_height,
            .use_low_perf = use_low_perf,
            .sub_core_grids = sub_core_grids,
        },
        TilizeDeviceOperation::tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::data_movement
