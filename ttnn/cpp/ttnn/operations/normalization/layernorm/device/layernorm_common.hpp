// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "layernorm_types.hpp"

namespace ttnn::prim {

// ROW_MAJOR activations are consumed as 1x32 faces (one row), matching
// matmul_decode::in0_tile_for_compute. TILE input uses its native tile.
inline tt::tt_metal::Tile compute_tile_for_layernorm(const Tensor& input) {
    if (input.layout() == Layout::ROW_MAJOR) {
        return tt::tt_metal::Tile({1, tt::constants::TILE_WIDTH}, false);
    }
    return input.tensor_spec().tile();
}

// Creates a program config from shard spec.
// - If shard_spec has value, creates a sharded config derived from it
// - Otherwise, returns a default interleaved config
LayerNormProgramConfig create_layernorm_program_config(
    const std::optional<tt::tt_metal::ShardSpec>& shard_spec, uint32_t tile_height = 32, uint32_t tile_width = 32);

}  // namespace ttnn::prim
