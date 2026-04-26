// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement::indexed_fill {

// Returns true if the program factory should pick the native CB-aliased fast path:
//   * input_a, batch_id and output are all L1
//   * input_a and output are HEIGHT_SHARDED with matching grid + matching shard shape
//   * the shard grid covers exactly B = input_a.padded_shape()[0] cores (one batch per core)
//   * input_a sharding is even (no leftover row/col)
//
// `input_b` is allowed to be in DRAM or interleaved (it is read via NoC accessor in the
// native reader); it does not need to match the shard geometry of input_a.
//
// Layout: applies to both ROW_MAJOR and TILE. Shard shape is interpreted in elements in
// either case, so the native reader's bulk local-NoC read of `page_size * pages_per_batch`
// bytes covers the per-core shard regardless of whether each page is a row or a tile.
bool is_native_indexed_fill_sharding(
    const TensorSpec& input_a_spec,
    const TensorSpec& input_b_spec,
    const TensorSpec& batch_id_spec,
    const tt::tt_metal::MemoryConfig& output_memory_config);

// Worker-grid selection priority:
//   1. explicit output shard grid (memory_config has a shard_spec) -> that grid
//   2. any sharded input (input_a > input_b > batch_id)            -> that input's shard grid
//   3. all worker cores of the device's first sub-device (default fallback)
CoreRangeSet get_indexed_fill_worker_grid(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& batch_id,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config);

// Returns true if the program factory should use the "shard-local" path for WIDTH_SHARDED or
// BLOCK_SHARDED input_a.  Conditions:
//   * input_a is WIDTH_SHARDED or BLOCK_SHARDED, L1
//   * output  is the same sharding layout, L1, with the same shard grid and shard shape
//   * input_b is either (a) the same sharding / grid / shape as input_a, or (b) INTERLEAVED
// In this mode every shard-grid core independently processes ALL batches in its local shard:
// page_size = shard_shape[1] * element_size, and the data CB is aliased to the output buffer.
bool is_shard_local_indexed_fill(
    const TensorSpec& input_a_spec,
    const TensorSpec& input_b_spec,
    const tt::tt_metal::MemoryConfig& output_memory_config);

// True iff the tensor is sharded and the shard shape does not evenly divide the padded shape.
bool is_uneven(const TensorSpec& t);

// Scale the input shard spec from `from_shape` to `to_shape` (preserves min 32 in each dim).
tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

// Build a default shard spec for `padded_out_shape` over the device's full compute grid.
tt::tt_metal::ShardSpec generate_shard_spec_all_cores(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, tt::tt_metal::TensorMemoryLayout memory_layout);

}  // namespace ttnn::operations::data_movement::indexed_fill
