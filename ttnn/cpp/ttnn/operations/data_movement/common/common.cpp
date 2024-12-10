// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/squeeze/squeeze.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {
ttnn::Tensor squeeze_to_le_4D(const ttnn::Tensor& tensor) {
    auto shape = tensor.get_shape();
    if (shape.rank() <= 4) {
        return tensor;
    } else {
        auto rank = shape.rank();
        auto squeezed = tensor;
        while (rank > 4) {
            squeezed = ttnn::squeeze(squeezed, 0);
            rank = squeezed.get_shape().rank();
        }
        return squeezed;
    }
};

ttnn::Tensor pad_to_tile_vol(
    uint8_t queue_id,
    const ttnn::Tensor& tensor,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config) {
    auto logical_shape = tensor.get_logical_shape();
    auto padded_shape = tensor.get_padded_shape();
    auto rank = tensor.get_shape().rank();
    if (padded_shape[-1] % tt::constants::TILE_WIDTH != 0 || padded_shape[-2] % tt::constants::TILE_HEIGHT != 0) {
        TT_ASSERT(rank >= 2, "rank of tensor to pad to tile must be at least 2.");

        auto padded_height = tt::round_up(padded_shape[-2], tt::constants::TILE_HEIGHT);
        auto padded_width = tt::round_up(padded_shape[-1], tt::constants::TILE_WIDTH);
        uint32_t num_non_hw_dims = rank - 2u;
        auto padding_vec = std::vector<std::pair<uint32_t, uint32_t>>(num_non_hw_dims, {0, 0});
        padding_vec.reserve(rank);
        padding_vec.emplace_back(0, padded_height - padded_shape[-2]);
        padding_vec.emplace_back(0, padded_width - padded_shape[-1]);

        constexpr bool pad_use_multicore = true;
        auto padded_output = ttnn::pad(queue_id, tensor, padding_vec, value, use_multicore, memory_config);
        TT_FATAL(
            padded_output.get_padded_shape()[-1] % tt::constants::TILE_WIDTH == 0 &&
                padded_output.get_padded_shape()[-2] % tt::constants::TILE_HEIGHT == 0,
            "pad_to_tile_vol: output tensor must be divisible by tile size");
        return padded_output;
    }
    return tensor;
}
uint32_t wrap_index(int index, int size) { return index < 0 ? size + index : index; }

ttnn::MemoryConfig create_sharded_memory_config(
    const ttnn::Shape& shape,
    const tt::tt_metal::CoreRangeSet& core_grid,
    const ShardStrategy& strategy,
    const tt::tt_metal::ShardOrientation& orientation,
    bool halo,
    bool use_height_and_width_as_shard_shape,
    const tt::tt_metal::Layout& layout) {
    auto is_tile_layout = layout == tt::tt_metal::Layout::TILE;

    auto rank = shape.rank();
    TT_FATAL(rank >= 2, "rank of tensor to shard must be at least 2.");

    auto tensor_memory_layout = ttnn::TensorMemoryLayout::BLOCK_SHARDED;
    if (strategy == ShardStrategy::WIDTH) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::WIDTH_SHARDED;
    } else if (strategy == ShardStrategy::HEIGHT) {
        tensor_memory_layout = ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
    }

    auto shard_orientation = orientation;
    auto shard_grid = core_grid;

    auto height = shape[-2];
    auto width = shape[-1];
    std::array<uint32_t, 2> shard_shape;

    if (use_height_and_width_as_shard_shape) {
        if (shard_orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR) {
            shard_shape = {height, width};
        } else if (shard_orientation == tt::tt_metal::ShardOrientation::COL_MAJOR) {
            shard_shape = {width, height};
        } else {
            TT_THROW("Invalid shard orientation");
        }
    } else {
        uint32_t batch_size = 1;
        for (int i = 0; i < rank - 2; i++) {
            batch_size *= shape[i];
        }

        auto tensor_height = batch_size * height;
        auto tensor_width = width;
        auto total_num_cores = shard_grid.num_cores();
        auto grid_size = shard_grid.bounding_box().grid_size();

        if (tensor_memory_layout == ttnn::TensorMemoryLayout::BLOCK_SHARDED) {
            TT_ASSERT(grid_size.y * grid_size.x == total_num_cores, "Invalid CoreRangeSet for block sharding strategy");

            if (shard_orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR) {
                auto tensor_height_padded =
                    is_tile_layout ? tt::round_up(tensor_height, grid_size.y * 32) : tensor_height;
                shard_shape = {tt::div_up(tensor_height_padded, grid_size.y), tt::div_up(tensor_width, grid_size.x)};
            } else if (shard_orientation == tt::tt_metal::ShardOrientation::COL_MAJOR) {
                auto tensor_height_padded =
                    is_tile_layout ? tt::round_up(tensor_height, grid_size.x * 32) : tensor_height;
                shard_shape = {tt::div_up(tensor_height_padded, grid_size.x), tt::div_up(tensor_width, grid_size.y)};
            } else {
                TT_THROW("Invalid shard orientation");
            }
        } else if (tensor_memory_layout == ttnn::TensorMemoryLayout::HEIGHT_SHARDED) {
            auto tensor_height_padded = is_tile_layout ? tt::round_up(tensor_height, total_num_cores) : tensor_height;
            shard_shape = {tt::div_up(tensor_height_padded, total_num_cores), tensor_width};
        } else if (tensor_memory_layout == ttnn::TensorMemoryLayout::WIDTH_SHARDED) {
            shard_shape = {tensor_height, tt::div_up(tensor_width, total_num_cores)};
        } else {
            TT_THROW("Invalid sharding scheme");
        }
    }

    if (is_tile_layout && shard_shape[0] % 32 != 0 && shard_shape[1] % 32 != 0) {
        TT_THROW("For sharding tiled tensors, the shard shape must fit neatly into tiles.");
    }

    auto shard_spec = tt::tt_metal::ShardSpec(shard_grid, shard_shape, shard_orientation, halo);
    return ttnn::MemoryConfig(tensor_memory_layout, ttnn::BufferType::L1, shard_spec);
}
}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
