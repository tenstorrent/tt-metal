// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::quasar::transpose_op {

// Native-sharded eligibility probe. Single-arg form: input only (pre-derivation). Two-arg form:
// also checks output side and (when both shard_specs are concrete) input/output grid match.
bool is_native_transpose_sharding(
    const TensorSpec& input_spec, const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

// Scale shard_spec from `from_shape` to `to_shape`; nullopt when scaling isn't exact.
std::optional<tt::tt_metal::ShardSpec> adjust_shard_spec_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape);

tt::tt_metal::ShardSpec generate_transpose_shard_spec(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, tt::tt_metal::TensorMemoryLayout memory_layout);

}  // namespace ttnn::operations::experimental::quasar::transpose_op
