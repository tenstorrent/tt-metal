// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include <tt-metalium/workload/legacy/host_api.hpp>

namespace ttnn::prim {

LayerNormProgramConfig create_layernorm_program_config(
    const std::optional<tt::tt_metal::ShardSpec>& shard_spec, uint32_t tile_height, uint32_t tile_width) {
    if (!shard_spec.has_value()) {
        return LayerNormDefaultProgramConfig{};
    }
    const auto& spec = shard_spec.value();
    const auto bbox = spec.grid.bounding_box();
    return LayerNormShardedMultiCoreProgramConfig{
        .compute_with_storage_grid_size =
            {bbox.end_coord.x - bbox.start_coord.x + 1, bbox.end_coord.y - bbox.start_coord.y + 1},
        .subblock_w = 1,
        .block_h = spec.shape[0] / tile_height,
        .block_w = spec.shape[1] / tile_width,
        .inplace = false,
    };
}

tt::tt_metal::ShardSpec interleaved_block_shard_spec(
    const Tensor& input, const LayerNormShardedMultiCoreProgramConfig& program_config) {
    const auto& tile = input.tensor_spec().tile();
    const uint32_t K = input.padded_shape()[-1];
    const uint32_t M = input.physical_volume() / K;
    const uint32_t block_h = program_config.block_h * tile.get_height();
    const uint32_t block_w = program_config.block_w * tile.get_width();
    TT_FATAL(
        block_h > 0 && block_w > 0 && M % block_h == 0 && K % block_w == 0,
        "Sharded layernorm over an interleaved input needs whole blocks: M {} and K {} must be multiples of block_h "
        "{} and block_w {} (in elements)",
        M,
        K,
        block_h,
        block_w);
    const uint32_t grid_x = K / block_w;
    const uint32_t grid_y = M / block_h;
    return tt::tt_metal::ShardSpec(
        CoreRangeSet(CoreRange({0, 0}, {grid_x - 1, grid_y - 1})),
        {block_h, block_w},
        tt::tt_metal::ShardOrientation::ROW_MAJOR);
}

}  // namespace ttnn::prim
