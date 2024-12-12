// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv2d {

struct Conv2dConfig {
    DataType dtype = DataType::BFLOAT16;
    DataType weights_dtype = DataType::BFLOAT16;
    string activation = "";
    uint32_t input_channels_alignment = 32;
    bool deallocate_activation = false;
    bool reallocate_halo_output = false;
    uint32_t act_block_h_override = 0; // This argument is ignored when shard_layout == WIDTH_SHARDED.
    uint32_t act_block_w_div = 1; // Amount by which the maximum possible act_block_width is divided. Max act_block_w = in_channels / (total_num_cores * TILE_WIDTH);
                                  // Ignored when shard_layout == HEIGHT_SHARDED or BLOCK_SHARDED
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
        "dtype",
        "weights_dtype",
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
            std::cref(this->dtype),
            std::cref(this->weights_dtype),
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


// TODO: Accept parallelization
enum class OptimizedConvOpParallelizationStrategy {
    MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE
};

struct OptimizedConvParallelizationConfig {
    CoreCoord grid_size; // (x,y)
    uint32_t num_cores_nhw = 1;
    uint32_t num_cores_c = 1;
    uint32_t per_core_out_matrix_height = 1;
    uint32_t per_core_out_matrix_width = 1;
    // std::size_t in0_block_w;
    // std::size_t out_subblock_h;
    // std::size_t out_subblock_w;
    // std::size_t per_core_M;
    // std::size_t per_core_N;

    CoreCoord get_grid_size() const {
        return this->grid_size;
    }
};

struct OptimizedConvBlockConfig {
    uint32_t act_block_h_ntiles;
    uint32_t act_block_w_ntiles;
    uint32_t out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles;
};

operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_new(const Tensor& a, const Tensor &b, const std::optional<const Tensor>& bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out, bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    bool use_shallow_conv_variant,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    Tensor& output,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding,
    bool use_non_tile_height);

// new micro op
struct OptimizedConvNew {
    OptimizedConvParallelizationConfig parallelization_config;
    OptimizedConvBlockConfig block_config;
    const sliding_window::SlidingWindowConfig& sliding_window_config;
    const uint32_t output_channels;
    const uint32_t groups;
    bool untilize_out, has_bias, fuse_relu;
    MemoryConfig memory_config;
    const DataType dtype;
    std::array<std::uint32_t, 4> input_tensor_shape; // For sharded input, input tensor shape is nonsense
    bool use_shallow_conv_variant;
    const DeviceComputeKernelConfig compute_kernel_config;
    bool enable_act_double_buffer;
    bool enable_weights_double_buffer;
    bool enable_split_reader;
    bool enable_subblock_padding;
    bool use_non_tile_height;
    uint32_t pre_op_l1_allocation_size_bytes;
    OptimizedConvNew(const sliding_window::SlidingWindowConfig& sliding_window_config,
        uint32_t output_channels, uint32_t groups,
        bool untile_out,
        bool has_bias, bool fuse_relu,
        const OptimizedConvParallelizationConfig& p_config,
        const OptimizedConvBlockConfig& b_config,
        MemoryConfig memory_config,
        DataType dtype,
        std::array<std::uint32_t, 4> input_tensor_shape, bool use_shallow_conv_variant,
        const DeviceComputeKernelConfig compute_kernel_config, bool enable_act_double_buffer, bool enable_weights_double_buffer, bool enable_split_reader, bool enable_subblock_padding, bool use_non_tile_height) :
            output_channels(output_channels),
            groups(groups),
            sliding_window_config(sliding_window_config),
            untilize_out(untile_out),
            has_bias(has_bias),
            fuse_relu(fuse_relu),
            parallelization_config(p_config),
            block_config(b_config),
            memory_config(memory_config),
            dtype(dtype), input_tensor_shape(input_tensor_shape),
            use_shallow_conv_variant(use_shallow_conv_variant),
            compute_kernel_config(compute_kernel_config),
            enable_act_double_buffer(enable_act_double_buffer),
            enable_weights_double_buffer(enable_weights_double_buffer),
            enable_split_reader(enable_split_reader),
            enable_subblock_padding(enable_subblock_padding),
            use_non_tile_height(use_non_tile_height) {}

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const;

    operation::OpPerformanceModel create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "parallelization_config",
        "block_config",
        "sliding_window_config",
        "output_channels",
        "untilize_out",
        "has_bias",
        "fuse_relu",
        "dtype",
        "input_tensor_shape",
        "use_shallow_conv_variant",
        "enable_act_double_buffer",
        "enable_weights_double_buffer",
        "enable_split_reader",
        "enable_subblock_padding");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->parallelization_config),
            std::cref(this->block_config),
            std::cref(this->sliding_window_config),
            std::cref(this->output_channels),
            std::cref(this->untilize_out),
            std::cref(this->has_bias),
            std::cref(this->fuse_relu),
            std::cref(this->dtype),
            std::cref(this->input_tensor_shape),
            std::cref(this->use_shallow_conv_variant),
            std::cref(this->enable_act_double_buffer),
            std::cref(this->enable_weights_double_buffer),
            std::cref(this->enable_split_reader),
            std::cref(this->enable_subblock_padding));
    }
};

Tensor optimized_conv_new(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out, bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    const MemoryConfig& memory_config,
    DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    bool use_shallow_conv_variant,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool enable_act_double_buffer = false,
    bool enable_weights_double_buffer = false,
    bool enable_split_reader = false,
    bool enable_subblock_padding = false,
    bool use_non_tile_height = false
);

std::pair<uint32_t,uint32_t> estimate_L1_usage(
    tt::ARCH arch, TensorMemoryLayout shard_layout,
    const DataType input_dtype, const DataType weights_dtype, const DataType output_dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const OptimizedConvBlockConfig& block_config, const OptimizedConvParallelizationConfig& pconfig,
    const Shape& input_shape, const Shape& weights_shape, const Shape& output_shape,
    uint32_t output_channels, uint32_t groups, std::array<uint32_t, 2> kernel_size,
    const Conv2dConfig& conv_config, bool enable_bias, bool use_non_tile_height
    );

}  // namespace conv2d

}  // namespace operations::conv

}  // namespace ttnn

namespace optimized_conv_op_utils {
using namespace tt;
using namespace tt::tt_metal;

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(
    const tt::tt_metal::LegacyShape& conv_activation_shape,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t num_cores_nhw,
    uint32_t act_block_h_ntiles);

} // optimized_conv_op_utils
