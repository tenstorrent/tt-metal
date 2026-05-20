// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NUKED-OP-STUB(conv2d): conv2d_utils.hpp was deleted by the conv2d nuke that
// prepares this branch for the matmul-helper op-gen evaluation. Pool internally
// consumed five helpers from that file. These stubs let pool COMPILE; they do
// NOT preserve runtime correctness. Pool tests will fail at runtime — that is
// the documented nuke-skill tradeoff ("surviving consumers do NOT need to be
// functionally correct — they only need to compile").

#pragma once

#include <cstdint>

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::conv {

namespace sliding_window = ttnn::operations::sliding_window;

inline sliding_window::ParallelConfig determine_parallel_config(
    tt::tt_metal::TensorMemoryLayout /*shard_layout*/,
    uint32_t /*batch_size*/,
    uint32_t /*input_channels*/,
    uint32_t /*output_height*/,
    uint32_t /*output_width*/,
    uint32_t /*output_channels*/,
    uint32_t /*channels_alignment*/,
    const CoreCoord& /*compute_grid_size*/,
    tt::tt_metal::ShardOrientation /*block_shard_orientation*/,
    bool /*enable_channels_padding*/,
    bool /*is_shard_height_tile_multiple*/,
    bool /*is_shard_width_tile_multiple*/,
    uint32_t /*act_block_h_override*/) {
    // TODO(nuked-op conv2d): real impl lived in conv2d_utils.cpp
    return sliding_window::ParallelConfig{};
}

inline uint32_t get_num_cores_channels_from_parallel_config(const sliding_window::ParallelConfig& /*pc*/) {
    // TODO(nuked-op conv2d)
    return 1;
}

inline uint32_t get_num_cores_nhw_from_parallel_config(const sliding_window::ParallelConfig& /*pc*/) {
    // TODO(nuked-op conv2d)
    return 1;
}

inline tt::tt_metal::MemoryConfig create_sharded_memory_config_from_parallel_config(
    const ttnn::Shape& /*shape*/, const sliding_window::ParallelConfig& /*pc*/, uint32_t /*shard_height*/) {
    // TODO(nuked-op conv2d)
    return tt::tt_metal::MemoryConfig{};
}

inline ttnn::Shape flatten_4d_shape(const ttnn::Shape& shape) {
    // TODO(nuked-op conv2d): real impl flattened NHWC -> (1, 1, N*H*W, C); passthrough here
    return shape;
}

}  // namespace ttnn::operations::conv
