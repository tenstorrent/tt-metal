// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TODO(nuked-op conv2d): conv2d (and conv1d / conv_transpose2d) were removed for eval.
// The real declarations lived in conv2d/conv2d_utils.hpp. A handful of surviving
// consumers (pool) used a small subset of those helpers. This stub provides
// passthrough / default-returning definitions for just that subset so the build
// stays green. The values are NOT correct — restore the real conv2d_utils.hpp and
// repoint consumers at it once conv2d is recreated.

#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::conv {

// TODO(nuked-op conv2d): restore real implementation.
inline sliding_window::ParallelConfig determine_parallel_config(
    tt::tt_metal::TensorMemoryLayout /*shard_layout*/,
    uint32_t /*batch_size*/,
    uint32_t /*input_channels*/,
    uint32_t /*output_height*/,
    uint32_t /*output_width*/,
    uint32_t /*output_channels*/,
    uint32_t /*input_channels_alignment*/,
    const CoreCoord& /*compute_grid_size*/,
    tt::tt_metal::ShardOrientation /*block_shard_orientation*/,
    bool /*enable_channels_padding*/,
    bool /*is_shard_height_tile_multiple*/ = true,
    bool /*is_shard_width_tile_multiple*/ = true,
    uint32_t /*act_block_h_override*/ = 0) {
    return sliding_window::ParallelConfig{};
}

// TODO(nuked-op conv2d): restore real implementation.
inline uint32_t get_num_cores_nhw_from_parallel_config(const sliding_window::ParallelConfig& /*pconfig*/) { return 1; }

// TODO(nuked-op conv2d): restore real implementation.
inline uint32_t get_num_cores_channels_from_parallel_config(const sliding_window::ParallelConfig& /*pconfig*/) {
    return 1;
}

// TODO(nuked-op conv2d): restore real implementation.
inline tt::tt_metal::MemoryConfig create_sharded_memory_config_from_parallel_config(
    const ttnn::Shape& /*tensor_shape*/,
    const sliding_window::ParallelConfig& /*parallel_config*/,
    uint32_t /*tile_size*/) {
    return tt::tt_metal::MemoryConfig{};
}

// TODO(nuked-op conv2d): restore real implementation (passthrough of input shape).
inline ttnn::Shape flatten_4d_shape(const ttnn::Shape& input_shape) { return input_shape; }

}  // namespace ttnn::operations::conv
