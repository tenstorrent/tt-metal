// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement::common {

// Orientation fallback: hint → input_orientation → ROW_MAJOR; `caller_tag` prefixes FATAL messages.
struct SynthesizeOutputShardSpecOpts {
    bool is_tile = true;
    std::optional<tt::tt_metal::ShardOrientation> orientation_hint = std::nullopt;
    std::optional<tt::tt_metal::ShardOrientation> input_orientation = std::nullopt;
    std::string_view caller_tag = "synthesize_output_shard_spec";
};

// Populated-shard CoreRangeSet for specless sharded outputs; BLOCK divisors track orientation for asymmetric grids.
tt::tt_metal::ShardSpec synthesize_output_shard_spec(
    const tt::tt_metal::CoreCoord& compute_grid_size,
    uint64_t tensor_height,
    uint64_t tensor_width,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    const SynthesizeOutputShardSpecOpts& opts = {});

// Convenience overload: flattens padded_out_shape into (product-of-leading, last-dim).
tt::tt_metal::ShardSpec synthesize_output_shard_spec(
    const tt::tt_metal::CoreCoord& compute_grid_size,
    const ttnn::Shape& padded_out_shape,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    const SynthesizeOutputShardSpecOpts& opts = {});

}  // namespace ttnn::operations::data_movement::common
