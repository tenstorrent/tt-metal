// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

void TensorSpec::validate_shard_spec_with_tensor_shape() const {
    const auto& memory_config = this->memory_config();
    if (memory_config.shard_spec.has_value()) {
        const auto& shape = logical_shape();

        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;

        uint32_t num_cores = shard_spec.num_cores();

        uint32_t total_height = shape.volume() / shape[-1];
        uint32_t total_width = shape[-1];
        if (memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(
                total_width == shard_shape[1],
                "Shard shape {} does not divide tensor shape {} correctly according to sharding scheme",
                shard_shape[1],
                total_width);
            uint32_t num_shards = div_up(total_height, shard_shape[0]);
            TT_FATAL(
                num_shards <= num_cores,
                "Number of shards along height {} must not exceed number of cores {}",
                num_shards,
                num_cores);
        } else if (memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            TT_FATAL(
                total_height == shard_shape[0],
                "Shard shape does not divide tensor shape correctly according to sharding scheme");
            uint32_t num_shards = div_up(total_width, shard_shape[1]);
            TT_FATAL(
                num_shards <= num_cores,
                "Number of shards along width {} must not exceed number of cores {}",
                num_shards,
                num_cores);
        } else if (memory_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                shard_spec.grid.ranges().size() == 1,
                "Shard grid must be one full rectangular grid for block sharded!");
            uint32_t num_shards_along_height = div_up(total_height, shard_shape[0]);
            uint32_t num_shards_along_width = div_up(total_width, shard_shape[1]);

            // Additionally check that number of cores along height and width matches shard grid
            const CoreCoord shard_grid = shard_spec.grid.bounding_box().grid_size();
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                TT_FATAL(
                    num_shards_along_height <= shard_grid.y,
                    "Number of shards along height {} must not exceed number of rows {} for row major orientation!",
                    num_shards_along_height,
                    shard_grid.y);
                TT_FATAL(
                    num_shards_along_width <= shard_grid.x,
                    "Number of shards along width {} must not exceed number of columns {} for row major orientation!",
                    num_shards_along_width,
                    shard_grid.x);
            } else {
                TT_FATAL(
                    num_shards_along_height <= shard_grid.x,
                    "Number of shards along height {} must not exceed number of columns {} for column major "
                    "orientation!",
                    num_shards_along_height,
                    shard_grid.x);
                TT_FATAL(
                    num_shards_along_width <= shard_grid.y,
                    "Number of shards along width {} must not exceed number of rows {} for column major orientation!",
                    num_shards_along_width,
                    shard_grid.y);
            }
        } else {
            TT_THROW("Unsupported sharding scheme");
        }
    }
    return;
}

}  // namespace tt::tt_metal
