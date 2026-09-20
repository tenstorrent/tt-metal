// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

// Generic NHW x C sharding helpers for sliding-window style ops (pooling, halo).
// They pick a core grid for a flattened [1, 1, N*H*W, C] activation and derive the
// matching L1 sharded MemoryConfig from a sliding_window::ParallelConfig.
namespace ttnn::operations::conv {

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor);
uint32_t find_closest_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor);
uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num, uint32_t start_divisor);
uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num1, uint32_t num2, uint32_t start_divisor);
uint32_t find_closest_largest_divisor_with_num_padding_and_mult(uint32_t num, uint32_t start_divisor, uint32_t mult);

uint32_t get_num_cores_nhw(
    const CoreRangeSet& cores, TensorMemoryLayout shard_layout, ShardOrientation shard_orientation);
uint32_t get_num_cores_channels(
    const CoreRangeSet& cores, TensorMemoryLayout shard_layout, ShardOrientation shard_orientation);
uint32_t get_num_cores_nhw_from_parallel_config(const sliding_window::ParallelConfig& pconfig);
uint32_t get_num_cores_channels_from_parallel_config(const sliding_window::ParallelConfig& pconfig);

sliding_window::ParallelConfig determine_parallel_config(
    TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    uint32_t input_channels_alignment,
    const CoreCoord& compute_grid_size,
    ShardOrientation block_shard_orientation,
    bool enable_channels_padding,
    bool is_shard_height_tile_multiple = true,
    bool is_shard_width_tile_multiple = true,
    uint32_t act_block_h_override = 0);

MemoryConfig create_sharded_memory_config_from_parallel_config(
    const ttnn::Shape& tensor_shape, const sliding_window::ParallelConfig& parallel_config, uint32_t tile_size);

ttnn::Shape flatten_4d_shape(const ttnn::Shape& input_shape);

}  // namespace ttnn::operations::conv
