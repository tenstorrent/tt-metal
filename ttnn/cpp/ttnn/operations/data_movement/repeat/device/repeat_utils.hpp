// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement::repeat {

// True when repeat can run natively on sharded L1 buffers (no strip/reshard round-trip).
bool is_native_repeat_sharding(
    const TensorSpec& input_spec,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt,
    int32_t repeat_dim = -1,
    uint32_t num_repeats = 1);

// Last-dim repeats are always locally contained on the same core.
bool is_replication_locally_contained(
    const tt::tt_metal::ShardSpec& input_shard_spec,
    const ttnn::Shape& input_padded_shape,
    int32_t repeat_dim,
    uint32_t num_repeats);

// Scale shard spec along repeat axis; nullopt if not exact.
std::optional<tt::tt_metal::ShardSpec> adjust_repeat_shard_spec_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec,
    const ttnn::Shape& from_shape,
    const ttnn::Shape& to_shape,
    int32_t repeat_dim,
    uint32_t num_repeats);

// Fresh shard spec over compute grid from post-repeat shape; nullopt if it won't fit.
std::optional<tt::tt_metal::ShardSpec> generate_repeat_shard_spec(
    const Tensor& input_tensor,
    const ttnn::Shape& padded_out_shape,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    std::optional<tt::tt_metal::ShardOrientation> orientation_hint = std::nullopt);

}  // namespace ttnn::operations::data_movement::repeat
