// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "layernorm_types.hpp"

namespace ttnn::prim {

// Creates a program config from shard spec.
// - If shard_spec has value, creates a sharded config derived from it
// - Otherwise, returns a default interleaved config
LayerNormProgramConfig create_layernorm_program_config(
    const std::optional<tt::tt_metal::ShardSpec>& shard_spec, uint32_t tile_height = 32, uint32_t tile_width = 32);

// The block grid an interleaved input is split over under a sharded program config, as the shard spec a block-sharded
// input of the same blocks would have: (Kt / block_w) x (Mt / block_h) cores from (0, 0), row major.
tt::tt_metal::ShardSpec interleaved_block_shard_spec(
    const Tensor& input, const LayerNormShardedMultiCoreProgramConfig& program_config);

}  // namespace ttnn::prim
