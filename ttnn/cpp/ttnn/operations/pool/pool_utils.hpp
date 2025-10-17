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
#include "ttnn/types.hpp"

namespace ttnn::operations::pool {

enum class Pool2DType {
    MAX_POOL2D = 0,
    AVG_POOL2D = 1,
};

struct AvgPoolConfig {
    uint32_t kernel_h{};
    uint32_t kernel_w{};
    uint32_t in_h{};
    uint32_t in_w{};
    uint32_t out_h{};
    uint32_t out_w{};
    uint32_t stride_h{};
    uint32_t stride_w{};
    bool ceil_mode{};
    uint32_t ceil_h{};
    uint32_t ceil_w{};
    bool count_include_pad{};
    uint32_t pad_t{};
    uint32_t pad_b{};
    uint32_t pad_l{};
    uint32_t pad_r{};
    uint32_t out_nhw_per_core{};
    std::optional<int32_t> divisor_override;
};

// DST Optimization Strategy for Pool Operations
enum class DSTOptimizationMode : uint32_t {
    SEQUENTIAL = 0,     // Original sequential processing (2 DST tiles)
    DUAL_POSITION = 1,  // 2-position parallel (4 DST tiles)
    QUAD_POSITION = 2   // 4-position parallel (8 DST tiles)
};

struct FactoryParameters {
    uint32_t multi_buffering_factor;
    bool split_reader;
    uint32_t nbytes;
    uint32_t index_nbytes;
    tt::DataFormat data_format;
    tt::DataFormat index_format;
    tt::DataFormat output_data_format;
    uint32_t in_ntiles_c;
    uint32_t out_ntiles_c;
    bool is_avg_pool;
    uint32_t max_rows_for_reduction;
    bool is_large_kernel;
    uint32_t MAX_TILES_PER_REDUCTION;
    bool is_wide_reduction;
    uint32_t num_tilized_rows;

    // DST Optimization fields
    DSTOptimizationMode dst_optimization_mode;
    bool l1_memory_constrained;
    uint32_t actual_l1_usage_bytes;
    uint32_t positions_per_iteration;  // 1, 2, or 4
    uint32_t dst_tiles_per_iteration;  // positions * channel_tiles
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
    std::optional<int32_t> divisor_override,
    bool return_indices,
    const Layout& output_layout,
    const DataType& output_dtype);

FactoryParameters get_factory_parameters(
    uint32_t num_shards_c,
    const DataType& input_dtype,
    const DataType& output_dtype,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_channels,
    Pool2DType pool_type,
    bool return_indices,
    const Layout& output_layout);

// DST Optimization functions
DSTOptimizationMode determine_dst_optimization_mode(
    uint32_t channel_tiles, uint32_t estimated_l1_usage, bool force_optimization = false);

uint32_t calculate_L1_usage_with_dst_optimization(
    const Tensor& input,
    uint32_t in_channels,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t ceil_pad_h,
    uint32_t ceil_pad_w,
    bool ceil_mode,
    bool return_indices,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t out_h,
    uint32_t out_w,
    const MemoryConfig& in_memory_config,
    const MemoryConfig& out_memory_config,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    const Layout& output_layout,
    const DataType& output_dtype,
    DSTOptimizationMode dst_mode);

bool is_dst_optimization_viable(uint32_t channel_tiles, uint32_t l1_usage_bytes, DSTOptimizationMode proposed_mode);

uint32_t calculate_L1_usage(
    const Tensor& input,
    uint32_t in_channels,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t ceil_pad_h,
    uint32_t ceil_pad_w,
    bool ceil_mode,
    bool return_indices,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t out_h,
    uint32_t out_w,
    const tt::tt_metal::MemoryConfig& input_memory,
    const tt::tt_metal::MemoryConfig& output_memory,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    const Layout& output_layout,
    const DataType& output_dtype);

// pool specific validations are done in validate_pool2d, but we want to validate basic inputs to ensure
// they are sensical to avoid problems in sliding window config, halo and other setup procedures
void validate_input_params(
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    uint32_t pad_top,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t pad_right,
    uint32_t dilation_h,
    uint32_t dilation_w,
    bool is_in_tiled);

}  // namespace ttnn::operations::pool
