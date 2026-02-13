// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::prim {

LayerNormProgramConfig create_layernorm_program_config(const std::optional<tt::tt_metal::ShardSpec>& shard_spec) {
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
