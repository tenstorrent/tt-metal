// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

enum class LayerNormType { LAYERNORM, RMSNORM };

enum class DistributedLayerNormStage { NOT_DISTRIBUTED, PRE_ALL_GATHER, POST_ALL_GATHER };

struct LayerNormDefaultProgramConfig {
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
};
struct LayerNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w{};
    std::size_t block_h{};
    std::size_t block_w{};
    bool inplace{};
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
};

using LayerNormProgramConfig = std::variant<LayerNormDefaultProgramConfig, LayerNormShardedMultiCoreProgramConfig>;

// Creates a program config from shard spec.
// - If shard_spec has value, creates a sharded config derived from it
// - Otherwise, returns a default interleaved config
inline LayerNormProgramConfig create_program_config(const std::optional<tt::tt_metal::ShardSpec>& shard_spec) {
    if (!shard_spec.has_value()) {
        return LayerNormDefaultProgramConfig{};
    }
    const auto& spec = shard_spec.value();
    const auto bbox = spec.grid.bounding_box();
    return LayerNormShardedMultiCoreProgramConfig{
        .compute_with_storage_grid_size =
            {bbox.end_coord.x - bbox.start_coord.x + 1, bbox.end_coord.y - bbox.start_coord.y + 1},
        .subblock_w = 1,
        .block_h = spec.shape[0] / tt::constants::TILE_HEIGHT,
        .block_w = spec.shape[1] / tt::constants::TILE_WIDTH,
        .inplace = false,
    };
}

}  // namespace ttnn::prim
