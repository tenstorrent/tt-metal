// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv2d {


namespace CMAKE_UNIQUE_NAMESPACE {
const uint32_t act_cb = tt::CBIndex::c_0;
const uint32_t weight_cb = tt::CBIndex::c_1;
const uint32_t bias_cb = tt::CBIndex::c_2;
const uint32_t sharded_act_cb = tt::CBIndex::c_3;
const uint32_t cb_for_reader_indices = tt::CBIndex::c_4;
const uint32_t cb_for_l1_array = tt::CBIndex::c_5;
const uint32_t act_cb_row_major_bfloat16 = tt::CBIndex::c_6;
const uint32_t act_cb_second_reader = tt::CBIndex::c_7;
const uint32_t matmul_partials_cb = tt::CBIndex::c_24;
const uint32_t tilize_mode_tilized_act_cb = tt::CBIndex::c_25;
const uint32_t untilize_mode_reblock_cb = tt::CBIndex::c_26;
const uint32_t out0_cb = tt::CBIndex::c_16;
const uint32_t temp_sum_cb = tt::CBIndex::c_27;
const uint32_t untilized_padded_out_cb = tt::CBIndex::c_28;
}  // namespace CMAKE_UNIQUE_NAMESPACE


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

operation::ProgramWithCallbacks multi_core_conv2d_impl(const Tensor& a, const Tensor &b, const std::optional<const Tensor>& bias,
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
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
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

std::tuple<CBHandle, CBHandle> create_CBs_for_depthwise_sharded_input(
    tt::tt_metal::Program& program,
    const Tensor& input,
    CoreRange core,
    uint32_t num_cb0_tiles,
    uint32_t num_cb1_tiles,
    uint32_t num_cb0_tilized_tiles,
    uint32_t num_output_tiles,
    uint32_t num_reblock_cb_tiles,
    uint32_t num_writer_output_tiles,
    bool untilize_out,
    tt::DataFormat act_df,
    tt::DataFormat weight_df,
    tt::DataFormat tilized_act_df,
    tt::DataFormat out_df,
    tt::DataFormat bias_df,
    bool weight_width_sliced,
    const Tensor& output,
    uint32_t bias_ntiles,
    bool with_bias,
    bool split_reader,
    bool fp32_dest_acc_en,
    bool packer_l1_acc_en);

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
