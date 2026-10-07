// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

// Builds a host tensor by copying whole tiles out of host source tensors, without unpacking them: the flat routed
// expert's weight layout is a pure tile permutation of the per-expert weights, and a block-float tile carries its own
// shared exponents, so moving packed tiles reuses the source quantization exactly and costs a memcpy per tile.
//
//   sources:   host TILE tensors, all of `dtype`, interleaved, distributed over the same mesh shards. Their tiles are
//              addressed in row-major tile order of each shard's padded 2D shape (higher dims folded into rows).
//   tile_map:  one entry per destination tile, in row-major tile order of the destination shard's padded 2D shape:
//              (source_index << 32) | source_tile, or -1 for a zero tile. The same map is applied to every shard, so
//              destination shard d reads shard d of each source.
//   shape / memory_config: the destination shard's logical shape and memory config (TILE layout, `dtype`).
//
// Returns a host tensor with the sources' mesh distribution.
ttnn::Tensor gather_host_tiles(
    const std::vector<ttnn::Tensor>& sources,
    const std::vector<int64_t>& tile_map,
    const ttnn::Shape& shape,
    const tt::tt_metal::MemoryConfig& memory_config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert
