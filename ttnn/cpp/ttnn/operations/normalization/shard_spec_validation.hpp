// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization::detail {

// Validates the standard shard-spec invariants shared by sharded normalization inputs
// (groupnorm / layernorm / softmax). Aborts via TT_FATAL on any violation.
//
// Checks performed:
//   * The input is sharded and has a populated shard_spec.
//   * The shard grid is non-empty and its bounding box fits within `program_grid_size`,
//     which in turn fits within the device grid.
//   * Shard dimensions are non-zero and shard height is a multiple of tile height.
//     Shard width is a multiple of tile width unless `require_shard_width_tile_aligned`
//     is false.
//   * The number of shards along H/W (derived from the layout) multiplied together
//     equals `shard_spec.grid.num_cores()`.
//   * Shards collectively cover the padded tensor with strictly less than one shard
//     of trailing pad along each axis (i.e. the last shard along an axis may be
//     partially filled, but a fully empty trailing shard is not allowed).
//
// `require_shard_width_tile_aligned` should be set to false for ops whose kernels are
// known to support shard widths that are not a multiple of the tile width
// (currently: groupnorm).
void validate_sharded_input(
    const ttnn::Tensor& tensor, const tt::tt_metal::CoreCoord& program_grid_size, bool require_shard_width_tile_aligned = true);

}  // namespace ttnn::operations::normalization::detail
