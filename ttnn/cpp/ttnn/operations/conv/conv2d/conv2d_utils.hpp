// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <unordered_set>

#include "ttnn/core.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/math.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "tt_metal/common/core_coord.hpp"

namespace ttnn {

namespace operations::conv {
using namespace conv2d;
using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;

struct Conv2dConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    DataType dtype = DataType::BFLOAT16;
    DataType weights_dtype = DataType::BFLOAT16;
    bool math_approx_mode_enabled = true;
    bool fp32_dest_acc_enabled = false;
    bool packer_l1_accum_enabled = false;
    string activation = "";
    uint32_t input_channels_alignment = 32;
    bool deallocate_activation = false;
    bool reallocate_halo_output = false;
    uint32_t act_block_h_override = 0; // This argument is ignored when shard_layout == WIDTH_SHARDED.
    uint32_t act_block_w_div = 1; //Amount by which the maximum possible act_block_width is divided. Max act_block_w = (in_channels * window_w * window_h)/total_num_cores;
                                  //Ignored when shard_layout == HEIGHT_SHARDED or BLOCK_SHARDED
    bool reshard_if_not_optimal = false; // if true, override_sharding_config should not be set to true
    bool override_sharding_config = false; // if true, reshard_if_not_optimal should not be set to true
    std::optional<TensorMemoryLayout> shard_layout;
    std::optional<CoreRangeSet> core_grid = std::nullopt; // used only if override_sharding_config is true
    bool transpose_shards = true; // used only if override_sharding_config is true and if height sharding is false
    Layout output_layout = Layout::TILE;
    bool enable_act_double_buffer = false;
    bool enable_weights_double_buffer = false; // Used on for block sharded convolutions
    bool enable_split_reader = false;
    bool enable_subblock_padding = false;
    static constexpr auto attribute_names = std::make_tuple(
        "math_fidelity",
        "dtype",
        "weights_dtype",
        "math_approx_mode_enabled",
        "fp32_dest_acc_enabled",
        "packer_l1_accum_enabled",
        "activation",
        "input_channels_alignment",
        "deallocate_activation",
        "reallocate_halo_output",
        "act_block_h_override",
        "act_block_w_div",
        "reshard_if_not_optimal",
        "override_sharding_config",
        "shard_layout",
        "core_grid",
        "transpose_shards",
        "output_layout",
        "enable_act_double_buffer",
        "enable_weights_double_buffer",
        "enable_split_reader",
        "enable_subblock_padding");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->math_fidelity),
            std::cref(this->dtype),
            std::cref(this->weights_dtype),
            std::cref(this->math_approx_mode_enabled),
            std::cref(this->fp32_dest_acc_enabled),
            std::cref(this->packer_l1_accum_enabled),
            std::cref(this->activation),
            std::cref(this->input_channels_alignment),
            std::cref(this->deallocate_activation),
            std::cref(this->reallocate_halo_output),
            std::cref(this->act_block_h_override),
            std::cref(this->act_block_w_div),
            std::cref(this->reshard_if_not_optimal),
            std::cref(this->override_sharding_config),
            std::cref(this->shard_layout),
            std::cref(this->core_grid),
            std::cref(this->transpose_shards),
            std::cref(this->output_layout),
            std::cref(this->enable_act_double_buffer),
            std::cref(this->enable_weights_double_buffer),
            std::cref(this->enable_split_reader),
            std::cref(this->enable_subblock_padding));
    }
};

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor);

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num, uint32_t start_divisor);

uint32_t find_closest_common_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor);

bool use_matmul_for_1x1_conv(
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& padding,
    const std::array<uint32_t, 2>& dilation,
    uint32_t groups);

sliding_window::ParallelConfig determine_parallel_config(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    const CoreCoord& compute_grid_size,
    ShardOrientation block_shard_orientation,
    bool is_out_tiled=true);

uint32_t get_num_cores_nhw_from_parallel_config(const sliding_window::ParallelConfig& pconfig);

uint32_t get_num_cores_channels_from_parallel_config(const sliding_window::ParallelConfig& pconfig);

MemoryConfig create_sharded_memory_config_from_parallel_config(const ttnn::Shape& tensor_shape, const sliding_window::ParallelConfig& parallel_config, uint32_t tile_size);

OptimizedConvParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(const MemoryConfig& conv_output_mem_config, uint32_t num_cores_nhw);

std::pair<uint32_t, uint32_t> determine_largest_subblock_size(uint32_t block_height, uint32_t block_width, bool fp32_accum);

ttnn::operations::matmul::MatmulProgramConfig determine_matmul_op_config_from_conv_op_config(
    OptimizedConvParallelizationConfig conv_parallelization_config,
    OptimizedConvBlockConfig conv_blocking_config,
    bool height_sharded,
    string activation,
    bool transpose_mcast,
    uint32_t grid_size_along_c);

OptimizedConvBlockConfig determine_per_core_conv_block_config(
    const sliding_window::ParallelConfig& parallel_config,
    const OptimizedConvParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t act_block_h_override,
    uint32_t act_block_w_div,
    uint32_t window_h,
    uint32_t window_w,
    bool fp32_accum,
    bool split_reader_enabled);

template<typename T>
std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool, bool> get_conv_padded_input_shape_and_mem_config(
    T * device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels);

OptimizedConvParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config, uint32_t num_cores_nhw, uint32_t num_cores_c);

void adjust_conv_op_config_for_auto_shard_if_necessary(
    bool is_mm_conv,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t weights_width,
    uint32_t input_width,
    const CoreCoord& compute_grid_size,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout,
    std::optional<const MemoryConfig> input_memory_config);

template <typename T>
std::tuple<ttnn::Tensor, sliding_window::ParallelConfig, sliding_window::ParallelConfig, bool, bool> shard_or_reshard_tensor_if_required(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv);

// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout(
    Tensor conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to tilized 2d matrix layout with special block height padding
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(
    Tensor conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to grouped layout with padded zeros
Tensor convert_conv_weight_tensor_to_grouped_layout(Tensor conv_weight_tensor, uint32_t num_groups, DataType output_dtype);

template <typename T>
OptimizedConvBlockConfig get_opt_block_config(
    bool mm_conv,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t batch_size,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    T *device,
    Layout input_tensor_layout,
    Conv2dConfig& conv_config);

} // namespace operations::conv
} // namespace ttnn
