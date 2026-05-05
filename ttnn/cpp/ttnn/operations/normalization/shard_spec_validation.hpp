// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization::detail {

// Validates standard shard invariants for a sharded normalization input:
//   1. shard grid fits within program grid; program grid fits within device grid; shard grid non-empty.
//   2. shard shape > 0; shard height tile-aligned; shard width tile-aligned iff `require_shard_width_tile_aligned`.
//   3. shard count matches the layout:
//        HEIGHT_SHARDED -> num_cores along height
//        WIDTH_SHARDED  -> num_cores along width
//        BLOCK_SHARDED  -> ceil(H_phys/shard_h) * ceil(W_phys/shard_w) == num_cores
//   4. shard-padded volume covers physical_volume() with trailing pad smaller than one shard along each axis
//      (no fully empty trailing shard).
//
// `require_shard_width_tile_aligned` defaults to true. Pass false for ops (e.g. groupnorm) whose kernels can handle
// shard widths that aren't multiples of tile width.
//
// Throws TT_FATAL on violation. Caller must already know the tensor is sharded.
void validate_sharded_input(
    const tt::tt_metal::Tensor& tensor,
    const CoreCoord& program_grid_size,
    bool require_shard_width_tile_aligned = true);

}  // namespace ttnn::operations::normalization::detail
