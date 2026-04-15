// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <optional>
#include <cstdint>

namespace ttnn::operations::data_movement::transpose {

bool is_uneven(const TensorSpec& t);

bool is_native_transpose_sharding(const TensorSpec& input_spec, const tt::tt_metal::MemoryConfig& output_memory_config);

struct TransposeShardSpecs {
    tt::tt_metal::ShardSpec input_shard_spec;
    tt::tt_metal::ShardSpec output_shard_spec;
};

std::optional<TransposeShardSpecs> get_transpose_shard_specs(
    const TensorSpec& input_spec, const TensorSpec& output_spec);

tt::tt_metal::ShardSpec adjust_shard_spec_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

tt::tt_metal::ShardSpec generate_transpose_shard_spec(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, tt::tt_metal::TensorMemoryLayout memory_layout);

CoreRangeSet get_transpose_worker_grid(
    const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_memory_config);

std::uint32_t* copy_transpose_common_runtime_args(const tt::tt_metal::Buffer& buffer, std::uint32_t* dst);

}  // namespace ttnn::operations::data_movement::transpose
