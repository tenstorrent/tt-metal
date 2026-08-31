// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <optional>

namespace ttnn::operations::unary {

/** True if native L1 sharding path can be used (input and output both L1, even sharding). */
bool is_native_L1_sharding(
    const tt::tt_metal::TensorSpec& input_spec, const tt::tt_metal::MemoryConfig& output_memory_config);

/** Shard spec for output when using native sharded path; nullopt if interleaved/fallback path. */
struct UnaryShardSpecs {
    tt::tt_metal::ShardSpec input_shard_spec;
    tt::tt_metal::ShardSpec output_shard_spec;
};

std::optional<UnaryShardSpecs> get_shard_specs(
    const tt::tt_metal::TensorSpec& input_spec, const tt::tt_metal::TensorSpec& output_spec);

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const tt::tt_metal::TensorSpec& tensor_spec);

bool is_uneven(const tt::tt_metal::TensorSpec& t);

CoreRangeSet get_worker_grid(
    const Tensor& input_tensor,
    const std::optional<Tensor>& output_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const tt::tt_metal::MemoryConfig& memory_config_actual);

/** Right-size an interleaved eltwise worker grid to the work size.
 *
 * The per-call cost of an interleaved eltwise program scales with the grid its kernels are
 * created on, not with the active work: every core of the worker grid gets kernels, and every
 * program-cache hit rewrites every core's runtime args. At a handful of tiles per core that
 * overhead dominates the op. This returns a contiguous sub-grid of `full_grid` sized
 * max(ceil(sqrt(num_tiles)), ceil(num_tiles/max_tiles_per_core)) — the measured optimum for
 * light eltwise ops (sqrt) with the per-core work capped so compute-heavy SFPU ops do not
 * regress (4 tiles/core for SFPU-chain work, 8 for plain binary FPU work) — rounded up to a
 * power of two so the number of distinct grids (each one is a separate program-cache entry,
 * since the worker grid is hashed) stays logarithmic. Returns `full_grid` unchanged once the
 * target reaches the full grid, so large-tensor calls keep today's grid and cache entry.
 * Shared by the unary and binary_ng interleaved default paths.
 */
CoreRangeSet right_size_worker_grid(
    const CoreRangeSet& full_grid, uint64_t num_tiles, uint32_t max_tiles_per_core);

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

/** Generate shard spec over all worker cores for a given output shape and memory layout. */
tt::tt_metal::ShardSpec generate_shard_spec_all_cores(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, tt::tt_metal::TensorMemoryLayout memory_layout);

}  // namespace ttnn::operations::unary
