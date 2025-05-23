// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <string>
#include <map>
#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {

namespace operations::pool {
enum class Pool2DType {
    MAX_POOL2D = 0,
    AVG_POOL2D = 1,
};

uint32_t get_bf16_pool_scalar(Pool2DType pool_type, uint32_t kernel_size_hw);
uint32_t get_bf16_pool_init_value(Pool2DType pool_type);
std::map<std::string, std::string> get_defines(Pool2DType pool_type);

std::optional<sliding_window::ParallelConfig> determine_parallel_config(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t channels,
    uint32_t output_height,
    uint32_t output_width,
    const CoreCoord& compute_grid_size,
    tt::tt_metal::ShardOrientation block_shard_orientation,
    bool enable_channels_padding,
    bool is_shard_height_tile_multiple = true,
    bool is_shard_width_tile_multiple = true,
    uint32_t act_block_h_override = 0);

std::optional<sliding_window::ParallelConfig> determine_pool_config_for_auto_shard(
    const Tensor& input_tensor,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t channels,
    Pool2DType pool_type);

uint32_t calculate_L1_usage(
    const Tensor& input,
    const uint32_t kernel_h,
    const uint32_t kernel_w,
    const uint32_t out_h,
    const uint32_t out_w,
    const MemoryConfig& input_memory,
    const MemoryConfig& output_memory,
    Pool2DType pool_type);

}  // namespace operations::pool
}  // namespace ttnn
