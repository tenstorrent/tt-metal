// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

// Compatibility types retained so surviving RMSNorm-family code compiles after
// the layernorm op family has been removed.
enum class LayerNormType { LAYERNORM, RMSNORM };
enum class DistributedLayerNormStage { NOT_DISTRIBUTED, PRE_ALL_GATHER, POST_ALL_GATHER };
enum class LayerNormDistributedType { LAYERNORM, RMSNORM };

struct LayerNormDefaultProgramConfig {
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
};

struct LayerNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size{};
    std::size_t subblock_w{};
    std::size_t block_h{};
    std::size_t block_w{};
    bool inplace{};
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
};

using LayerNormProgramConfig = std::variant<LayerNormDefaultProgramConfig, LayerNormShardedMultiCoreProgramConfig>;

inline LayerNormProgramConfig create_layernorm_program_config(
    const std::optional<tt::tt_metal::ShardSpec>& shard_spec, uint32_t tile_height = 32, uint32_t tile_width = 32) {
    if (!shard_spec.has_value()) {
        return LayerNormDefaultProgramConfig{};
    }
    (void)shard_spec;
    (void)tile_height;
    (void)tile_width;
    LayerNormShardedMultiCoreProgramConfig cfg{};
    cfg.compute_with_storage_grid_size = CoreCoord{0, 0};
    cfg.subblock_w = 1u;
    cfg.block_h = 1u;
    cfg.block_w = 1u;
    cfg.inplace = false;
    return cfg;
}

}  // namespace ttnn::prim
