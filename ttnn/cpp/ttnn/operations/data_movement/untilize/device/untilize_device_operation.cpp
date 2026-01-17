// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "factories/untilize_single_core_program_factory.hpp"
#include "factories/untilize_multi_core_sub_core_grids_program_factory.hpp"
#include "factories/untilize_multi_core_block_program_factory.hpp"
#include "factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.hpp"
#include "factories/untilize_multi_core_parallelize_column_program_factory.hpp"
#include "factories/untilize_multi_core_program_factory.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/common/constants.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

uint32_t get_pf_type(bool output_is_sharded, const Tensor& tensor) {
    auto* device = tensor.device();
    uint32_t max_l1_size =
        (device->l1_size_per_core() / 2) - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(input_cb_data_format);
    // Determine the max number of tiles that can be in any CB at a given time (1 input CB + 1 output CB = 2 total CBs)
    uint32_t max_tiles_per_cb = max_l1_size / (2 * single_tile_size);

    // TODO : currently multi_core parallelization on column only works for single tile height tensors.
    // Need to debug this to work on wide tensors that are higher than a single tile
    const auto& tile_shape = tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tensor_width = tensor.padded_shape()[-1];
    uint32_t tensor_height = tensor.physical_volume() / tensor_width;
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    uint32_t num_tiles_per_row = tensor_width / tile_width;
    uint32_t num_tiles_per_col = tensor_height / tile_height;

    // If the input is interleaved and an entire row of tiles can't fit in a CB at once
    if (!tensor.is_sharded() && num_tiles_per_row > max_tiles_per_cb) {
        // If the output is also interleaved and the tensor is only a single tile high, we can
        // parellize the work column wise. Otherwise we have to resort to the single core implementation,
        // as the current default multi core implementation processes an entire row of tiles at once.
        if (!output_is_sharded && num_tiles_per_col == 1) {
            return 0;
        }
        return 1;
    }
    return 2;
}

void UntilizeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void UntilizeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor_a = tensor_args.input;

    uint32_t tensor_width = input_tensor_a.padded_shape()[-1];
    uint32_t tensor_height = input_tensor_a.physical_volume() / tensor_width;

    bool input_is_sharded = input_tensor_a.is_sharded();
    bool output_is_sharded = operation_attributes.output_mem_config.is_sharded();

    BufferType input_buffer_type = input_tensor_a.memory_config().buffer_type();
    BufferType output_buffer_type = operation_attributes.output_mem_config.buffer_type();

    TensorMemoryLayout input_memory_layout = input_tensor_a.memory_config().memory_layout();
    TensorMemoryLayout output_memory_layout = operation_attributes.output_mem_config.memory_layout();

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    // Input must be in valid tile layout
    TT_FATAL(tensor_width % TILE_WIDTH == 0, "Width must be evenly divisible into tiles");
    TT_FATAL(tensor_height % TILE_HEIGHT == 0, "Height must be evenly divisible into tiles");

    // Special conditions for sub_core_grids special case
    if (operation_attributes.sub_core_grids.has_value()) {
        TT_FATAL(
            input_memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Input memory layout must be interleaved when sub_core_grid argument provided");
        TT_FATAL(
            output_memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Output memory layout must be interleaved when sub_core_grid argument provided");
        TT_FATAL(
            operation_attributes.use_multicore,
            "sub_core_grid implementation only supported when use_multicore flag argument is set to true");
    }

    // If input is sharded, then the shard shape must be in multiples of tiles
    if (input_is_sharded) {
        std::array<uint32_t, 2> input_shard_shape = input_tensor_a.shard_spec().value().shape;
        uint32_t input_shard_width = input_shard_shape[1];
        uint32_t input_shard_height = input_shard_shape[0];
        TT_FATAL(
            input_shard_width % TILE_WIDTH == 0,
            "Input shard width {} must be a multiple of tile width",
            input_shard_width);
        TT_FATAL(
            input_shard_height % TILE_HEIGHT == 0,
            "Input shard height {} must be a multiple of tile height",
            input_shard_height);
    }

    // We don't support input or output uneven sharding for the single core implementation
    if (!operation_attributes.use_multicore) {
        // Check for input uneven sharding
        if (input_is_sharded) {
            std::array<uint32_t, 2> input_shard_shape = input_tensor_a.shard_spec().value().shape;
            uint32_t input_shard_width = input_shard_shape[1];
            uint32_t input_shard_height = input_shard_shape[0];
            TT_FATAL(
                tensor_width % input_shard_width == 0,
                "Uneven input shard width {} for tensor width {} not supported for single core implementation",
                input_shard_width,
                tensor_width);
            TT_FATAL(
                tensor_height % input_shard_height == 0,
                "Uneven input shard height {} for tensor height {} not supported for single core implementation",
                input_shard_height,
                tensor_height);
        }
        // Check for output uneven sharding
        if (output_is_sharded) {
            std::array<uint32_t, 2> output_shard_shape =
                operation_attributes.output_mem_config.shard_spec().value().shape;
            uint32_t output_shard_width = output_shard_shape[1];
            uint32_t output_shard_height = output_shard_shape[0];
            TT_FATAL(
                tensor_width % output_shard_width == 0,
                "Uneven output shard width {} for tensor width {} not supported for single core implementation",
                output_shard_width,
                tensor_width);
            TT_FATAL(
                tensor_height % output_shard_height == 0,
                "Uneven output shard height {} for tensor height {} not supported for single core implementation",
                output_shard_height,
                tensor_height);
        }
    }

    // We always support uneven input sharding for the multicore implementation. Uneven output sharding is only
    // supported if the input and output memory layouts are identical (i.e. height->height, width->width, block->block)
    // and the input and output shard specs are identical. Otherwise uneven output sharding is not supported.
    if (output_is_sharded) {
        std::array<uint32_t, 2> output_shard_shape = operation_attributes.output_mem_config.shard_spec().value().shape;
        uint32_t output_shard_width = output_shard_shape[1];
        uint32_t output_shard_height = output_shard_shape[0];

        bool output_is_uneven_sharded_width_wise = tensor_width % output_shard_width != 0;
        bool output_is_uneven_sharded_height_wise = tensor_height % output_shard_height != 0;
        if (output_is_uneven_sharded_width_wise || output_is_uneven_sharded_height_wise) {
            TT_FATAL(
                input_memory_layout == output_memory_layout,
                "Input and output memory layouts must be identical if output is uneven sharded");
            TT_FATAL(
                input_tensor_a.shard_spec().value() == operation_attributes.output_mem_config.shard_spec().value(),
                "Input and output shard specs must be identical if output is uneven sharded");
        }
    }

    // Multicore implementation doesn't support input DRAM sharding
    if (operation_attributes.use_multicore && input_is_sharded) {
        TT_FATAL(input_buffer_type == BufferType::L1, "Multicore implementation doesn't support DRAM sharding");
    }

    // We don't support output DRAM block sharding
    if (output_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(output_buffer_type == BufferType::L1, "We don't support DRAM block sharding");
    }

    // Pack untilize is what allows uint32/int32 support, so if it is not enabled, we do not support uint32/int32
    if (!operation_attributes.use_pack_untilize) {
        TT_FATAL(
            input_tensor_a.dtype() != DataType::UINT32 && input_tensor_a.dtype() != DataType::INT32,
            "Pack untilize must be enabled to support uint32/int32 data types");
    }
}

UntilizeDeviceOperation::spec_return_value_t UntilizeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input;
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();

    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

UntilizeDeviceOperation::tensor_return_value_t UntilizeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

UntilizeDeviceOperation::program_factory_t UntilizeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input;
    const auto& output_tensor = operation_attributes.output_mem_config;

    bool input_is_sharded = input_tensor_a.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    BufferType input_buffer_type = input_tensor_a.memory_config().buffer_type();
    BufferType output_buffer_type = output_tensor.buffer_type();

    TensorMemoryLayout input_memory_layout = input_tensor_a.memory_config().memory_layout();
    TensorMemoryLayout output_memory_layout = output_tensor.memory_layout();

    if (!operation_attributes.use_multicore) {
        // Single core implementation
        return program::UntilizeSingleCoreProgramFactory{};
    }
    if (operation_attributes.sub_core_grids.has_value()) {
        // If sub_core_grids parameter is provided, use custom sub_core_grid implementation instead
        // of the standard multicore implementation or the block multicore implementation.
        // Note that this implementation does not support sharding, which is enforced in validate().
        return program::UntilizeMultiCoreSubCoreGridsProgramFactory{};
    }
    if (!operation_attributes.enough_space_height && !input_is_sharded && !output_is_sharded) {
        // Optimized special case implementation, only supported when neither input or output is sharded
        return program::UntilizeMultiCoreBlockProgramFactory{};
    }
    if (input_is_sharded && output_is_sharded && input_buffer_type == BufferType::L1 &&
        output_buffer_type == BufferType::L1 && input_memory_layout == output_memory_layout &&
        input_tensor_a.shard_spec() == output_tensor.shard_spec()) {
        // Optimized special case implementation for when both input and output are sharded, both are located in L1,
        // have identical memory layouts (i.e. height->height, width->width, block->block), and have identical shard
        // specs
        return program::UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory{};
    }

    uint32_t tensor_width = input_tensor_a.padded_shape()[-1];
    uint32_t tensor_height = input_tensor_a.physical_volume() / tensor_width;

    const auto& tile_shape = input_tensor_a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    uint32_t num_tiles_per_row = tensor_width / tile_width;
    uint32_t num_tiles_per_col = tensor_height / tile_height;

    auto grid_size = input_tensor_a.device()->compute_with_storage_grid_size();

    size_t grid_area = grid_size.x * grid_size.y;
    auto [num_compute_cores, nblocks_per_core] = compute_ncores(grid_area, num_tiles_per_col);

    constexpr uint32_t threshold_row_block = 32;
    if (!input_is_sharded and !output_is_sharded) {
        if (num_tiles_per_row > threshold_row_block and
            (num_tiles_per_col > threshold_row_block or num_tiles_per_row > num_tiles_per_col)) {
            uint32_t num_blocks_block = (input_tensor_a.padded_shape()[-1] * input_tensor_a.padded_shape()[-2]) /
                                        (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
            auto ncores_wh = compute_ncores_wh(grid_area, num_blocks_block, num_tiles_per_row, num_tiles_per_col);
            if (num_compute_cores < ncores_wh.ncores) {
                return program::UntilizeMultiCoreBlockProgramFactory{};
            }
        }
    }
    // TODO : currently multi_core parallelization on column only works for single tile height tensors.
    // Need to debug this to work on wide tensors that are higher than a single tile
    auto pf_option = get_pf_type(output_is_sharded, input_tensor_a);
    if (pf_option == 0) {
        return program::UntilizeMultiCoreParallelizeColumnProgramFactory{};
    }
    if (pf_option == 1) {
        return program::UntilizeSingleCoreProgramFactory{};
    }
    // Default multi core implementation
    return program::UntilizeMultiCoreProgramFactory{};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<UntilizeDeviceOperation::tensor_return_value_t>
UntilizeDeviceOperation::create_op_performance_model(
    const UntilizeDeviceOperation::operation_attributes_t& /*op_attr*/,
    const UntilizeDeviceOperation::tensor_args_t& inputs,
    tensor_return_value_t& output) {
    const auto& input_tensor = inputs.input;
    const auto& output_tensor = output;
    uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    uint32_t single_tile_size = tile_width * tile_height * input_tensor.element_size();
    uint32_t num_tiles =
        std::ceil(static_cast<float>(input_tensor.physical_volume()) / static_cast<float>(single_tile_size));
    int compute_cycles = 0;
    const int max_tiles_per_row = 8;
    const int latency_untilize = 390;      // measured latency for untilize_block
    const int latency_pack_untilize = 80;  // measured latency for pack_untilize_block
    if (std::ceil(static_cast<float>(input_tensor.padded_shape()[-1]) / static_cast<float>(tile_width)) <=
        max_tiles_per_row) {
        compute_cycles = num_tiles * latency_pack_untilize;
    } else {
        compute_cycles = num_tiles * latency_untilize;
    }

    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor, false, compute_cycles);
    tt::tt_metal::operation::OpPerformanceModelGeneral<UntilizeDeviceOperation::tensor_return_value_t> result(
        {input_tensor}, output_tensor, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::UntilizeDeviceOperation::tensor_return_value_t untilize(
    const Tensor& input,
    tt::tt_metal::MemoryConfig output_mem_config,
    bool use_multicore,
    bool use_pack_untilize,
    bool fp32_dest_acc_en,
    std::optional<CoreRangeSet> sub_core_grids,
    bool enough_space_width,
    bool enough_space_height,
    uint32_t pf_type) {
    using OperationType = ttnn::operations::data_movement::UntilizeDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = std::move(output_mem_config),
            .use_multicore = use_multicore,
            .use_pack_untilize = use_pack_untilize,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .sub_core_grids = std::move(sub_core_grids),
            .enough_space_width = enough_space_width,
            .enough_space_height = enough_space_height,
            .pf_type = pf_type},
        OperationType::tensor_args_t{.input = input});
}
}  // namespace ttnn::prim
