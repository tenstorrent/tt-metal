// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn {

namespace operations::conv {
using namespace conv2d;
using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor);

uint32_t find_closest_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor);

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num, uint32_t start_divisor);

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num1, uint32_t num2, uint32_t start_divisor);

bool use_matmul_for_1x1_conv(
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 4>& padding,
    const std::array<uint32_t, 2>& dilation,
    uint32_t groups,
    const Conv2dConfig& conv_config);

bool is_1d_conv(uint32_t kernel_width, uint32_t image_width);

bool is_1d_deptwise_conv(
    uint32_t groups,
    uint32_t input_channels,
    uint32_t output_channels,
    uint32_t kernel_width,
    uint32_t image_width,
    bool has_bias);

sliding_window::ParallelConfig determine_parallel_config(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    const CoreCoord& compute_grid_size,
    tt::tt_metal::ShardOrientation block_shard_orientation,
    bool enable_channels_padding,
    bool is_shard_height_tile_multiple = true,
    bool is_shard_width_tile_multiple = true,
    uint32_t act_block_h_override = 0);

sliding_window::ParallelConfig determine_output_parallel_config(
    const sliding_window::ParallelConfig& input_parallel_config,
    const CoreCoord& compute_grid_size,
    uint32_t out_channels,
    bool is_mm_conv);

std::tuple<uint32_t, uint32_t> calculate_output_image_size(
    std::array<uint32_t, 2> input_image_size,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding,
    std::array<uint32_t, 2> dilation);

uint32_t get_num_cores_nhw_from_parallel_config(const sliding_window::ParallelConfig& pconfig);

uint32_t get_num_cores_channels_from_parallel_config(const sliding_window::ParallelConfig& pconfig);

MemoryConfig create_sharded_memory_config_from_parallel_config(
    const ttnn::Shape& tensor_shape, const sliding_window::ParallelConfig& parallel_config, uint32_t tile_size);

OptimizedConvParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config, uint32_t num_cores_nhw, uint32_t num_cores_c);

ttnn::operations::matmul::MatmulProgramConfig determine_matmul_op_config_from_conv_op_config(
    OptimizedConvParallelizationConfig conv_parallelization_config,
    OptimizedConvBlockConfig conv_blocking_config,
    bool height_sharded,
    const string& activation,
    bool transpose_mcast,
    uint32_t grid_size_along_c);

OptimizedConvBlockConfig determine_per_core_conv_block_config(
    const sliding_window::ParallelConfig& parallel_config,
    const OptimizedConvParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t padded_output_height_ntiles_per_core,
    uint32_t act_block_h_override,
    uint32_t act_block_w_div,
    uint32_t window_h,
    uint32_t window_w,
    bool fp32_accum,
    bool split_reader_enabled);

std::tuple<OptimizedConvParallelizationConfig, OptimizedConvBlockConfig, MemoryConfig> get_conv_configs(
    const Conv2dConfig& conv_config,
    const DeviceComputeKernelConfig& compute_config,
    const sliding_window::ParallelConfig& input_parallel_config,
    const sliding_window::ParallelConfig& output_parallel_config,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t output_height,
    uint32_t output_width,
    std::array<uint32_t, 2> kernel_size,
    const CoreCoord& compute_grid);

template <typename T>
static std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv);

template <typename DeviceType>
DeviceComputeKernelConfig get_conv_default_compute_kernel_config(DeviceType* device);

Conv2dConfig determine_conv_config_for_auto_shard(
    const Conv2dConfig& conv_config_,
    bool is_mm_conv,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t weights_width,
    uint32_t input_height,
    uint32_t input_width,
    const CoreCoord& compute_grid_size,
    Layout input_tensor_layout,
    std::optional<const MemoryConfig> input_memory_config,
    const std::array<uint32_t, 2>& kernel_size,
    const uint32_t groups,
    const bool enable_bias,
    const DeviceComputeKernelConfig& compute_config);

ttnn::Shape flatten_4d_shape(const ttnn::Shape& input_shape);

template <typename T>
std::tuple<ttnn::Tensor, sliding_window::ParallelConfig, sliding_window::ParallelConfig>
shard_or_reshard_tensor_if_required(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv,
    bool auto_shard);

std::ostream& operator<<(std::ostream& os, const Conv2dConfig& config);

}  // namespace operations::conv
}  // namespace ttnn
