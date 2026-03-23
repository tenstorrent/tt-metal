// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <optional>

namespace ttnn::operations::unary_ng {

/** True if native L1 sharding path can be used (input and output both L1, even sharding). */
bool is_native_L1_sharding(const TensorSpec& input_spec, const MemoryConfig& output_memory_config);

/** Shard spec for output when using native sharded path; nullopt if interleaved/fallback path. */
struct UnaryShardSpecs {
    tt::tt_metal::ShardSpec input_shard_spec;
    tt::tt_metal::ShardSpec output_shard_spec;
};

std::optional<UnaryShardSpecs> get_shard_specs(const TensorSpec& input_spec, const TensorSpec& output_spec);

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const TensorSpec& tensor_spec);

bool is_uneven(const TensorSpec& t);

CoreRangeSet get_worker_grid(
    const Tensor& input_tensor,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const MemoryConfig& memory_config_actual);

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

/** Generate shard spec over all worker cores for a given output shape and memory layout. */
tt::tt_metal::ShardSpec generate_shard_spec_all_cores(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, tt::tt_metal::TensorMemoryLayout memory_layout);

}  // namespace ttnn::operations::unary_ng
