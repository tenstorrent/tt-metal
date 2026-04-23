// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

namespace ttnn::prim {
tt::tt_metal::ReduceOpParallelizationStrategy get_parallelization_strategy(
    const tt::tt_metal::Tensor& input_tensor, tt::tt_metal::ReduceOpDim reduce_dim) {
    uint32_t num_tiles = input_tensor.physical_volume() / input_tensor.tensor_spec().tile().get_tile_hw();
    if (reduce_dim == tt::tt_metal::ReduceOpDim::H) {
        return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_H;
    }
    if (reduce_dim == tt::tt_metal::ReduceOpDim::W) {
        return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_W;
    }
    if (reduce_dim == tt::tt_metal::ReduceOpDim::HW) {
        if (num_tiles > 1) {
            return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_HW;
        }
        return tt::tt_metal::ReduceOpParallelizationStrategy::SINGLE_CORE_HW;
    }
    TT_THROW("Unsupported reduce dim");
}

tt::tt_metal::TensorSpec build_reduce_output_tensor_spec(
    const tt::tt_metal::Shape& output_shape,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::MemoryConfig& input_mem_config,
    tt::tt_metal::ReduceOpDim reduce_dim) {
    using namespace tt::tt_metal;

    TensorSpec tensor_spec(
        output_shape,
        TensorLayout(output_dtype, PageConfig(Layout::TILE), MemoryConfig(output_mem_config.buffer_type())));

    TensorMemoryLayout mem_layout = output_mem_config.memory_layout();

    if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED || mem_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
        mem_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        // Grid and orientation are identical in both spec formats (nd_shard_spec and shard_spec)
        // when both are populated. Pick whichever is available from the output config,
        // falling back to the input tensor's shard spec for backward compatibility.
        const auto& nd = output_mem_config.nd_shard_spec();
        const auto& legacy = output_mem_config.shard_spec();
        const auto& input_nd = input_mem_config.nd_shard_spec();
        const auto& input_legacy = input_mem_config.shard_spec();
        auto get_grid_and_orientation = [&]() -> std::pair<const CoreRangeSet&, ShardOrientation> {
            if (nd) {
                return {nd->grid, nd->orientation};
            }
            if (legacy) {
                return {legacy->grid, legacy->orientation};
            }
            if (input_nd) {
                return {input_nd->grid, input_nd->orientation};
            }
            if (input_legacy) {
                return {input_legacy->grid, input_legacy->orientation};
            }
            TT_THROW(
                "Sharded memory layout {} requires either nd_shard_spec or shard_spec to be set "
                "on the output memory config or the input tensor",
                mem_layout);
        };
        const auto& [grid, orientation] = get_grid_and_orientation();

        // For width/height/block sharding modes, the output shard shape is fully determined
        // by the output physical shape and the core grid. Just delegate to the
        // appropriate TensorSpec builder.
        if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            return tensor_spec.width_sharded(grid, orientation);
        }
        if (mem_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            return tensor_spec.height_sharded(grid, orientation);
        }
        TT_FATAL(
            grid.ranges().size() == 1,
            "Block sharding requires a single CoreRange, got {} ranges",
            grid.ranges().size());
        return tensor_spec.block_sharded(grid.bounding_box(), orientation);
    }

    // ND sharding: adjust per-logical-dimension shard shape for reduced dims.
    // Fall back to the input tensor's nd_shard_spec when the output config omits it.
    if (mem_layout == TensorMemoryLayout::ND_SHARDED) {
        const auto& nd_shard_spec = output_mem_config.nd_shard_spec();
        const auto& input_nd_shard_spec = input_mem_config.nd_shard_spec();
        TT_FATAL(
            nd_shard_spec.has_value() || input_nd_shard_spec.has_value(),
            "ND_SHARDED memory layout requires nd_shard_spec to be set "
            "on the output memory config or the input tensor");
        auto nd_shard_spec_copy = nd_shard_spec.has_value() ? *nd_shard_spec : *input_nd_shard_spec;
        if (reduce_dim == ReduceOpDim::W || reduce_dim == ReduceOpDim::HW) {
            nd_shard_spec_copy.shard_shape[-1] = 1;
        }
        if ((reduce_dim == ReduceOpDim::H || reduce_dim == ReduceOpDim::HW) &&
            nd_shard_spec_copy.shard_shape.rank() > 1) {
            nd_shard_spec_copy.shard_shape[-2] = 1;
        }
        return tensor_spec.sharded(std::move(nd_shard_spec_copy), TensorSpec::ShardShapeAlignment::REQUIRED);
    }

    // Guard against unexpected new memory layouts.
    TT_FATAL(mem_layout == TensorMemoryLayout::INTERLEAVED, "Unexpected memory layout: {}", mem_layout);
    // Interleaved tensor: tensor_spec already has everything we need.
    return tensor_spec;
}
}  // namespace ttnn::prim
