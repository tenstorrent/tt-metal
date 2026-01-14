// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::untilize_with_unpadding {

UntilizeWithUnpaddingDeviceOperation::program_factory_t UntilizeWithUnpaddingDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_tensor.memory_config().is_sharded()) {
        TT_FATAL(
            !operation_attributes.sub_core_grids.has_value(),
            "Sharded untilize does not support sub core grid specification");
        return program::UntilizeWithUnpaddingMultiCoreShardedProgramFactory{};
    }
    if (!operation_attributes.use_multicore) {
        return program::UntilizeWithUnpaddingSingleCoreProgramFactory{};
    }
    if (!operation_attributes.enough_space_height) {
        return program::UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory{};
    }
    const auto& a = tensor_args.input_tensor;
    const auto& input_shape = a.padded_shape();
    auto* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid =
        operation_attributes.sub_core_grids.has_value() ? operation_attributes.sub_core_grids.value() : default_grid;

    uint32_t num_blocks = input_shape[-1] == 0 ? 0 : a.physical_volume() / input_shape[-1] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    uint32_t num_tiles_per_col = a.padded_shape()[-2] / tt::constants::TILE_HEIGHT;

    size_t grid_area = available_grid.num_cores();
    auto [ncores, nblocks_per_core] = compute_ncores(grid_area, num_blocks);
    constexpr uint32_t threshold_row_block = 32;
    if (num_tiles_per_row > threshold_row_block &&
        (num_tiles_per_col > threshold_row_block || num_tiles_per_row > num_tiles_per_col)) {
        uint32_t num_blocks_block =
            (a.padded_shape()[-1] * a.padded_shape()[-2]) / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);

        auto ncores_wh = compute_ncores_wh(grid_area, num_blocks_block, num_tiles_per_row, num_tiles_per_col);
        if (ncores < ncores_wh.ncores) {
            return program::UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory{};
        }
    }
    return program::UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory{};
}

void UntilizeWithUnpaddingDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void UntilizeWithUnpaddingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_FATAL(
        input_tensor_a.physical_volume() % tt::constants::TILE_HW == 0,
        "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
        input_tensor_a.physical_volume(),
        tt::constants::TILE_HW);

    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                input_tensor_a.shard_spec().value().grid.ranges().size() == 1,
                "Expected single grid range and got {}",
                input_tensor_a.shard_spec().value().grid.ranges().size());
            TT_FATAL(
                operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Output memory config layout must be INTERLEAVED for block sharded input but got {}",
                operation_attributes.output_mem_config.memory_layout());
            TT_FATAL(
                input_tensor_a.physical_volume() /
                        (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                    1,
                "Can only write unbatched output interleaved");
        } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            if (operation_attributes.output_mem_config.is_sharded()) {
                TT_FATAL(
                    operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                    "Output memory config layout must be HEIGHT_SHARDED when output is sharded but got {}",
                    operation_attributes.output_mem_config.memory_layout());
            }
            // What else?
        } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            auto output_shape = compute_output_specs(operation_attributes, tensor_args).padded_shape();
            for (uint32_t i = 0; i < output_shape.rank() - 2; i++) {
                TT_FATAL(
                    input_tensor_a.padded_shape()[i] == output_shape[i],
                    "Input tensor padded shape[{}] ({}) must equal output shape[{}] ({})",
                    i,
                    input_tensor_a.padded_shape()[i],
                    i,
                    output_shape[i]);
            }
            if (operation_attributes.output_mem_config.is_sharded()) {
                TT_FATAL(
                    operation_attributes.output_mem_config.memory_layout() ==
                        input_tensor_a.memory_config().memory_layout(),
                    "Output memory config layout ({}) must match input tensor memory layout ({})",
                    operation_attributes.output_mem_config.memory_layout(),
                    input_tensor_a.memory_config().memory_layout());
                TT_FATAL(
                    input_tensor_a.padded_shape()[-1] == output_shape[-1] ||
                        (tt::div_up(output_shape[-1], input_tensor_a.shard_spec().value().shape[1]) ==
                         input_tensor_a.shard_spec().value().grid.num_cores()),
                    "Input tensor width ({}) must equal output width ({}) or output width / shard width must equal num "
                    "cores",
                    input_tensor_a.padded_shape()[-1],
                    output_shape[-1]);
            } else {
                TT_FATAL(
                    operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                    "Output memory config layout must be INTERLEAVED but got {}",
                    operation_attributes.output_mem_config.memory_layout());
                TT_FATAL(
                    input_tensor_a.physical_volume() /
                            (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                        1,
                    "Can only write unbatched output interleaved");
                TT_FATAL(
                    input_tensor_a.padded_shape()[-1] - output_shape[-1] < input_tensor_a.shard_spec().value().shape[1],
                    "Input tensor width difference ({}) must be less than shard width ({})",
                    input_tensor_a.padded_shape()[-1] - output_shape[-1],
                    input_tensor_a.shard_spec().value().shape[1]);
            }
        } else {
            TT_THROW("Unsupported sharding scheme");
        }
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

    // Pack untilize is what allows uint32/int32 support, so if it is not enabled, we do not support uint32/int32
    if (!operation_attributes.use_pack_untilize) {
        TT_FATAL(
            input_tensor_a.dtype() != DataType::UINT32 && input_tensor_a.dtype() != DataType::INT32,
            "Pack untilize must be enabled to support uint32/int32 data types");
    }
}

UntilizeWithUnpaddingDeviceOperation::spec_return_value_t UntilizeWithUnpaddingDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    SmallVector<uint32_t> out_shape;
    const auto& input_tensor_a = tensor_args.input_tensor;
    size_t rank = input_tensor_a.logical_shape().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(operation_attributes.output_tensor_end[i] + 1);
    }
    Shape output_shape(std::move(out_shape));

    DataType output_dtype = input_tensor_a.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor_a.dtype();
    if (input_tensor_a.memory_config().is_sharded() && operation_attributes.output_mem_config.is_sharded()) {
        uint32_t fused_height = output_shape.volume() / output_shape[-1];
        uint32_t num_cores = input_tensor_a.shard_spec().value().num_cores();
        std::array<uint32_t, 2> shard_shape{};
        ShardSpec shard_spec = input_tensor_a.shard_spec().value();
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            const auto tile = input_tensor_a.tensor_spec().tile();
            uint32_t tile_height = tile.get_height();
            uint32_t shard_idx0 = tt::round_up(tt::div_up(fused_height, num_cores), tile_height);
            shard_shape = {shard_idx0, output_shape[-1]};
        } else {
            shard_shape = {fused_height, shard_spec.shape[1]};
        }
        shard_spec.shape = shard_shape;
        auto mem_config = operation_attributes.output_mem_config.with_shard_spec(shard_spec);
        return TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), mem_config));
    }

    return TensorSpec(
        output_shape,
        TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), operation_attributes.output_mem_config));
}

UntilizeWithUnpaddingDeviceOperation::tensor_return_value_t UntilizeWithUnpaddingDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>
UntilizeWithUnpaddingDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    uint32_t single_tile_size = tile_width * tile_height * input_tensor.element_size();
    uint32_t num_tiles = std::ceil((float)input_tensor.physical_volume() / (float)single_tile_size);
    int compute_cycles = 0;
    const int max_tiles_per_row = 8;
    const int latency_untilize = 390;      // measured latency for untilize_block
    const int latency_pack_untilize = 80;  // measured latency for pack_untilize_block
    if (std::ceil((float)input_tensor.padded_shape()[-1] / (float)tile_width) <= max_tiles_per_row) {
        compute_cycles = num_tiles * latency_pack_untilize;
    } else {
        compute_cycles = num_tiles * latency_untilize;
    }
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor, false, compute_cycles);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, output_tensor, ideal_dev_clock_cycles);
    return result;
}
}  // namespace ttnn::operations::data_movement::untilize_with_unpadding

namespace ttnn::prim {
ttnn::operations::data_movement::untilize_with_unpadding::UntilizeWithUnpaddingDeviceOperation::tensor_return_value_t
untilize_with_unpadding(
    const Tensor& input_tensor,
    const ttnn::Shape& output_tensor_end,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    bool use_multicore,
    bool use_pack_untilize,
    bool fp32_dest_acc_en,
    bool enough_space_width,
    bool enough_space_height,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType =
        ttnn::operations::data_movement::untilize_with_unpadding::UntilizeWithUnpaddingDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_tensor_end = output_tensor_end,
            .output_mem_config = output_mem_config.value_or(input_tensor.memory_config()),
            .use_multicore = use_multicore,
            .use_pack_untilize = use_pack_untilize,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .enough_space_width = enough_space_width,
            .enough_space_height = enough_space_height,
            .sub_core_grids = sub_core_grids},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim
