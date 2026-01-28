// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::conv {
using ttnn::prim::Conv2dBlockConfig;
using ttnn::prim::Conv2dConfig;
using ttnn::prim::Conv2dInputs;
using ttnn::prim::Conv2dParallelizationConfig;
using ttnn::prim::Conv2dParams;
using ttnn::prim::Conv2dSliceConfig;
using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor);

uint32_t find_closest_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor);

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num, uint32_t start_divisor);

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num1, uint32_t num2, uint32_t start_divisor);

uint32_t get_input_channels_alignment(
    TensorMemoryLayout input_tensor_memory_layout,
    Layout input_tensor_layout,
    bool sliced_op,
    bool is_mm_conv,
    const std::optional<MemoryConfig>& input_memory_config);

CoreCoord get_output_compute_grid_size(
    const CoreCoord& device_compute_grid_size,
    const Conv2dConfig& conv_config,
    const sliding_window::ParallelConfig& input_parallel_config);

bool use_matmul_for_1x1_conv(
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 4>& padding,
    const std::array<uint32_t, 2>& dilation,
    uint32_t groups,
    const Conv2dConfig& conv_config);

bool is_1d_conv(uint32_t kernel_height, uint32_t image_height);

bool is_1d_depthwise_conv(
    uint32_t groups,
    uint32_t input_channels,
    uint32_t output_channels,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t image_height,
    bool has_bias);

struct SkipMcast {
    bool skip_activation_mcast;
    bool skip_weights_mcast;
};

SkipMcast conv_skip_mcast(const Conv2dParallelizationConfig& parallelization_config, TensorMemoryLayout memory_layout);

sliding_window::ParallelConfig determine_parallel_config(
    TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    uint32_t input_channels_alignment,
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
    tt::tt_metal::ShardOrientation block_shard_orientation,
    bool is_mm_conv);

std::tuple<uint32_t, uint32_t> calculate_output_image_size(
    std::array<uint32_t, 2> input_image_size,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding,
    std::array<uint32_t, 2> dilation);

std::tuple<uint32_t, uint32_t> calculate_ct2d_output_image_size(
    std::array<uint32_t, 2> input_image_size,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation);

uint32_t get_num_cores_nhw(
    const CoreRangeSet& cores, TensorMemoryLayout shard_layout, ShardOrientation shard_orientation);
uint32_t get_num_cores_nhw_from_parallel_config(const sliding_window::ParallelConfig& pconfig);

uint32_t get_num_cores_channels(
    const CoreRangeSet& cores, TensorMemoryLayout shard_layout, ShardOrientation shard_orientation);
uint32_t get_num_cores_channels_from_parallel_config(const sliding_window::ParallelConfig& pconfig);

MemoryConfig create_sharded_memory_config_from_parallel_config(
    const ttnn::Shape& tensor_shape, const sliding_window::ParallelConfig& parallel_config, uint32_t tile_size);

Conv2dParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config,
    uint32_t num_cores_nhw,
    uint32_t num_cores_c_in,
    uint32_t num_cores_c_out);

ttnn::operations::matmul::MatmulProgramConfig determine_matmul_op_config_from_conv_op_config(
    Conv2dParallelizationConfig conv_parallelization_config,
    Conv2dBlockConfig conv_blocking_config,
    bool height_sharded,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    bool transpose_mcast,
    uint32_t grid_size_along_c);

Conv2dBlockConfig determine_per_core_conv_block_config(
    const sliding_window::ParallelConfig& parallel_config,
    const Conv2dParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t padded_output_height_ntiles_per_core,
    uint32_t act_block_h_override,
    uint32_t act_block_w_div,
    uint32_t window_h,
    uint32_t window_w,
    uint32_t output_width,
    bool fp32_accum,
    bool full_inner_dim,
    bool enable_activation_reuse = false,
    bool is_1d_depthwise_conv = false);

std::tuple<Conv2dParallelizationConfig, Conv2dBlockConfig, MemoryConfig> get_conv_configs(
    const Conv2dConfig& conv_config,
    const DeviceComputeKernelConfig& compute_config,
    const sliding_window::ParallelConfig& input_parallel_config,
    const sliding_window::ParallelConfig& output_parallel_config,
    uint32_t in_channels_padded,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t output_height,
    uint32_t output_width,
    std::array<uint32_t, 2> kernel_size,
    const CoreCoord& compute_grid,
    bool is_1d_depthwise_conv = false);

std::tuple<ttnn::Shape, ttnn::MemoryConfig> determine_input_memory_config(
    TensorMemoryLayout shard_layout,
    ShardOrientation block_shard_orientation,
    uint32_t batch_size,
    ttnn::Shape input_tensor_shape,
    ttnn::Shape output_tensor_shape,
    bool is_mm_conv,
    CoreCoord compute_grid_size,
    Layout input_tensor_layout,
    BufferType input_tensor_buffer_type,
    const std::optional<sliding_window::ParallelConfig>& input_tensor_parallel_config = std::nullopt,
    std::optional<uint32_t> act_block_h_override = std::nullopt,
    bool enable_channels_padding = true,
    bool is_shard_height_tile_multiple = true,
    bool is_shard_width_tile_multiple = true);

DeviceComputeKernelConfig get_conv_default_compute_kernel_config(
    MeshDevice* device, DataType input_dtype, DataType weight_dtype);

struct core_count_and_size {
    uint32_t core_count{};
    uint32_t halo_input_size{};
    uint32_t halo_output_size{};
    uint32_t conv_op_size{};
    uint32_t total_size{};
    Conv2dConfig conv_config;
};

core_count_and_size calculate_L1_usage_for_conv_op(
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t output_height,
    uint32_t output_width,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 4>& padding,
    const std::array<uint32_t, 2>& dilation,
    uint32_t groups,
    bool enable_bias,
    DataType input_datatype,
    DataType output_datatype,
    Layout input_layout,
    CoreCoord compute_grid_size,
    bool is_mm_conv,
    TensorMemoryLayout shard_layout,
    DeviceComputeKernelConfig compute_config,
    const Conv2dConfig& conv_config_in,
    const std::optional<ttnn::MemoryConfig>& _halo_input_memory_config = std::nullopt);

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
    Layout input_layout,
    tt::tt_metal::DataType input_datatype,
    tt::tt_metal::DataType output_datatype,
    std::optional<const MemoryConfig> input_memory_config,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& dilation,
    const std::array<uint32_t, 4>& padding,
    uint32_t groups,
    bool enable_bias,
    const DeviceComputeKernelConfig& compute_config);

ttnn::Shape flatten_4d_shape(const ttnn::Shape& input_shape);

std::tuple<ttnn::Tensor, sliding_window::ParallelConfig, sliding_window::ParallelConfig>
shard_or_reshard_tensor_if_required(
    MeshDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv,
    bool auto_shard);

bool auto_enable_kernel_folding(
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    const DataType& input_dtype,
    std::optional<bool> enable_folding_,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2>& kernel_size,
    std::array<uint32_t, 2>& stride,
    std::array<uint32_t, 2>& dilation,
    std::array<uint32_t, 4>& padding_n4);

Tensor fold_input_tensor_if_required(
    const ttnn::Tensor& input_tensor,
    MeshDevice* device,
    uint32_t& batch_size,
    uint32_t& input_height,
    uint32_t& input_width,
    uint32_t& in_channels,
    std::array<uint32_t, 2>& kernel_size,
    std::array<uint32_t, 2>& stride,
    std::array<uint32_t, 2>& dilation,
    std::array<uint32_t, 4>& padding_n4,
    bool& mm_conv,
    Conv2dConfig& conv_config);

ttnn::Tensor fold_tensor(
    const ttnn::Tensor& tensor,
    MeshDevice* device,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 4> padding_n4,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t in_channels);

struct KernelStrideFoldingResult {
    uint32_t input_height;
    uint32_t input_width;
    uint32_t in_channels;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 4> padding_n4;
    bool mm_conv;
};

KernelStrideFoldingResult compute_kernel_stride_folding_params(
    uint32_t input_height,
    uint32_t input_width,
    uint32_t in_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding_n4,
    const Conv2dConfig& conv_config);
std::ostream& operator<<(std::ostream& os, const Conv2dConfig& config);

struct ConvDRAMParamters {
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t batch_size;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t output_height;
    uint32_t output_width;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 4> padding_n4;
    std::array<uint32_t, 2> dilation;
    uint32_t groups;
    Conv2dConfig conv_config;
    DeviceComputeKernelConfig compute_kernel_config;
    CoreCoord compute_grid;
    DataType weights_datatype;
    DataType input_datatype;
    DataType output_datatype;
    Layout input_layout;
    bool enable_bias;
    bool mm_conv;

    bool operator<(const ConvDRAMParamters& other) const;

    static constexpr auto attribute_names = std::make_tuple(
        "in_channels",
        "out_channels",
        "batch_size",
        "input_height",
        "input_width",
        "output_height",
        "output_width",
        "kernel_size",
        "stride",
        "padding_n4",
        "dilation",
        "groups",
        "conv_config",
        "compute_kernel_config",
        "compute_grid",
        "weights_datatype",
        "input_datatype",
        "output_datatype",
        "input_layout",
        "enable_bias",
        "mm_conv");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->in_channels),
            std::cref(this->out_channels),
            std::cref(this->batch_size),
            std::cref(this->input_height),
            std::cref(this->input_width),
            std::cref(this->output_height),
            std::cref(this->output_width),
            std::cref(this->kernel_size),
            std::cref(this->stride),
            std::cref(this->padding_n4),
            std::cref(this->dilation),
            std::cref(this->groups),
            std::cref(this->conv_config),
            std::cref(this->compute_kernel_config),
            std::cref(this->compute_grid),
            std::cref(this->weights_datatype),
            std::cref(this->input_datatype),
            std::cref(this->output_datatype),
            std::cref(this->input_layout),
            std::cref(this->enable_bias),
            std::cref(this->mm_conv));
    }
};

void tilize_with_optional_deallocation(Tensor& input_tensor_on_device, bool deallocate);

// Enum to represent the execution path for conv2d operations
enum class Conv2dExecutionPath {
    L1,   // Execute conv2d using L1 memory
    DRAM  // Execute conv2d using DRAM slicing
};

// Helper function to determine which conv2d execution path to take based on
// slice configuration and input tensor properties
Conv2dExecutionPath determine_conv2d_execution_path(
    const ttnn::Tensor& input_tensor, const std::optional<const Conv2dSliceConfig>& slice_config);
Conv2dExecutionPath determine_conv2d_execution_path(
    bool input_is_in_L1, const std::optional<const Conv2dSliceConfig>& slice_config);
}  // namespace ttnn::operations::conv
