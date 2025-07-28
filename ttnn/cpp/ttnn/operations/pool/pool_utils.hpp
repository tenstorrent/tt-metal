// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <string>
#include <map>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::pool {

enum class Pool2DType {
    MAX_POOL2D = 0,
    AVG_POOL2D = 1,
};

struct AvgPoolConfig {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t stride_h;
    uint32_t stride_w;
    bool ceil_mode;
    uint32_t ceil_h;
    uint32_t ceil_w;
    bool count_include_pad;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t out_nhw_per_core;
    std::optional<int32_t> divisor_override;
};

uint32_t get_bf16_pool_scalar(
    Pool2DType pool_type, uint32_t kernel_h, uint32_t kernel_w, std::optional<int32_t> divisor_override);
uint32_t get_bf16_pool_init_value(Pool2DType pool_type);
std::map<std::string, std::string> get_defines(Pool2DType pool_type);

bool is_pool_op_one_scalar_per_core(
    Pool2DType pool_type,
    bool ceil_mode,
    uint32_t ceil_h,
    uint32_t ceil_w,
    bool count_include_pad,
    uint32_t pad_h,
    uint32_t pad_w,
    std::optional<int32_t> divisor_override);

std::optional<sliding_window::ParallelConfig> determine_valid_parallel_config(
    tt::tt_metal::TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t channels,
    uint32_t output_height,
    uint32_t output_width,
    const CoreCoord& compute_grid_size,
    tt::tt_metal::ShardOrientation block_shard_orientation,
    bool enable_channels_padding,
    bool is_shard_height_tile_multiple = false,
    bool is_shard_width_tile_multiple = false,
    uint32_t act_block_h_override = 0);

std::optional<sliding_window::ParallelConfig> determine_pool_config_for_auto_shard(
    const Tensor& input_tensor,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t channels,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override);

std::tuple<uint32_t, bool, uint32_t, tt::DataFormat, uint32_t, bool, uint32_t, bool, uint32_t, bool>
get_factory_parameters(
    uint32_t num_shards_c, const Tensor& input, uint32_t kernel_h, uint32_t kernel_w, Pool2DType pool_type);

uint32_t calculate_L1_usage(
    const Tensor& input,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t ceil_pad_h,
    uint32_t ceil_pad_w,
    bool ceil_mode,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t out_h,
    uint32_t out_w,
    const tt::tt_metal::MemoryConfig& input_memory,
    const tt::tt_metal::MemoryConfig& output_memory,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override);

}  // namespace ttnn::operations::pool
