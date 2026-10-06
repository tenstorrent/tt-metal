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

// `is_tile`: drives alignment (32 TILE, 1 RM) + non-sharded-axis round_up; reflect the *output* layout.
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

// Strict: reject on no valid tensor_width divisor (caller falls back).
// Lenient: never nullopt for RM — tile-pads shard_width (WIDTH_SHARDED) or returns `spec` unchanged.
enum class RmPageAlignmentMode : uint8_t { Strict, Lenient };

// RM WIDTH_SHARDED L1-page-alignment adjuster: shrinks num_cores to a valid tensor_width divisor.
// Non-RM inputs pass through unchanged (TILE lands aligned by construction).
std::optional<tt::tt_metal::ShardSpec> shrink_shard_for_rm_page_alignment(
    const tt::tt_metal::ShardSpec& spec,
    tt::tt_metal::Layout input_layout,
    uint32_t element_size_bytes,
    uint64_t tensor_width,
    const tt::tt_metal::CoreCoord& compute_grid_size,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    RmPageAlignmentMode mode = RmPageAlignmentMode::Strict);

}  // namespace ttnn::operations::data_movement::common
