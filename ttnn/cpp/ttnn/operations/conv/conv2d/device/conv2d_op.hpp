// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv2d {

struct Conv2dConfig {
    // If set, the weights & bias tensors will be converted to this dtype after preprocessing.
    // prepare_conv_bias needs this to always be set to the same dtype as the weights.
    std::optional<tt::tt_metal::DataType> weights_dtype = std::nullopt;

    // Either "relu" or ""
    std::string activation = "";

    // If user tensor will be deallocated if it's on device.
    bool deallocate_activation = false;

    // If true && dellocate_activation is true, then after halo device op is done,
    // the output tensor of halo will be reallocated.
    bool reallocate_halo_output = true;

    // Has to be a multiple of 32.
    //  Smaller -> Smaller CBs, Lower L1 Usage, Lower perf.
    uint32_t act_block_h_override = 0;

    // Amount by which the maximum possible act_block_width is divided.
    // Max act_block_w = in_channels / (total_num_cores * TILE_WIDTH);
    // Ignored when shard_layout == HEIGHT_SHARDED or BLOCK_SHARDED
    // Only useful when in_channels > 2048.
    uint32_t act_block_w_div = 1;

    // Only considered when input is already sharded in L1.
    //  if reshard_if_not_optimal is true, override_sharding_config should not be set to true
    bool reshard_if_not_optimal = false;

    // if override_sharding_config is true, reshard_if_not_optimal should not be set to true
    bool override_sharding_config = false;

    std::optional<tt::tt_metal::TensorMemoryLayout> shard_layout;

    // used only if override_sharding_config is true
    std::optional<CoreRangeSet> core_grid = std::nullopt;

    // used only if override_sharding_config is true and shard_layout is set to BLOCK_SHARDED
    bool transpose_shards = false;

    // Useful when output is BFLOAT16.
    // BFLOAT8 is always Tile layout.
    tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE;

    // Doubles the size of the CBs for activation.
    // Increased perf, but increased L1 usage.
    bool enable_act_double_buffer = false;

    // Doubles the size of the CBs for weights.
    // Increased perf, but increased L1 usage.
    bool enable_weights_double_buffer = false;

    // Only for height sharding.
    // Increases perf if op is reader bound. Act_block_h should be >= 64, if true
    bool enable_split_reader = false;

    bool enable_subblock_padding = false;

    // Re-use input tensor storage when creating output tensor
    bool in_place = false;

    // ==================== EXPERIMENTAL FEATURES ====================
    // Features in this section are under development.
    // Use with caution.

    // Kernel Stride Folding (Issue: #22378)
    // Enables tensor folding optimization where:
    // - Input tensor (NHWC) is reshaped to (N, H/stride[0], W/stride[1], C * stride[0] * stride[1])
    // - Weight tensor (OC, IC, kernel[0], kernel[1]) is reshaped and permuted to (1, 1, IC * kernel[0] * kernel[1], OC)
    // Currently only applied when strides match kernel dimensions
    bool enable_kernel_stride_folding = false;
    // ===============================================================

    static constexpr auto attribute_names = std::make_tuple(
        "weights_dtype",
        "activation",
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
        "enable_subblock_padding",
        "in_place",
        "enable_kernel_stride_folding");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->weights_dtype),
            std::cref(this->activation),
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
            std::cref(this->enable_subblock_padding),
            std::cref(this->in_place),
            std::cref(this->enable_kernel_stride_folding));
    }
};

struct Conv2dSliceConfig {
    // Determines the dimension along which the input & output tensors are sliced.
    // Slices based on [N, H, W, C] shape.
    // Using width slicing is more efficient as it reduces memory usage. This is because the overlap of data between
    // cores is minimized in width slicing, reducing the size of the Halo output. If the Height & Width dimensions are
    // similar, then use Width slicing. Use Height slicing if the Height dimension is significantly larger than the
    // Width dimension.
    enum class SliceType : bool { HEIGHT, WIDTH };
    SliceType slice_type = SliceType::WIDTH;

    // Number of slices that the output tensor should be divided into.
    uint32_t num_slices = 0;
};

// TODO: Accept parallelization
enum class OptimizedConvOpParallelizationStrategy { MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE };

struct OptimizedConvParallelizationConfig {
    CoreCoord grid_size;  // (x,y)
    uint32_t num_cores_nhw = 1;
    uint32_t num_cores_c = 1;
    uint32_t per_core_out_matrix_height_ntile = 1;
    uint32_t per_core_out_matrix_width_ntile = 1;

    CoreCoord get_grid_size() const { return this->grid_size; }
};

struct OptimizedConvBlockConfig {
    uint32_t act_block_h_ntiles;
    uint32_t act_block_w_ntiles;
    uint32_t out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles;
};

tt::tt_metal::operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_new(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    tt::tt_metal::DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    Tensor& output,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding);

// new micro op
struct OptimizedConvNew {
    OptimizedConvParallelizationConfig parallelization_config;
    OptimizedConvBlockConfig block_config;
    const sliding_window::SlidingWindowConfig& sliding_window_config;
    const uint32_t output_channels;
    const uint32_t groups;
    bool untilize_out, has_bias;
    std::string activation = "";
    tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
    std::array<std::uint32_t, 4> input_tensor_shape;  // For sharded input, input tensor shape is nonsense
    const DeviceComputeKernelConfig compute_kernel_config;
    bool enable_act_double_buffer;
    bool enable_weights_double_buffer;
    bool enable_split_reader;
    bool enable_subblock_padding;
    uint32_t pre_op_l1_allocation_size_bytes;
    OptimizedConvNew(
        const sliding_window::SlidingWindowConfig& sliding_window_config,
        uint32_t output_channels,
        uint32_t groups,
        bool untile_out,
        bool has_bias,
        std::string activation,
        const OptimizedConvParallelizationConfig& p_config,
        const OptimizedConvBlockConfig& b_config,
        tt::tt_metal::MemoryConfig memory_config,
        tt::tt_metal::DataType dtype,
        std::array<std::uint32_t, 4> input_tensor_shape,
        const DeviceComputeKernelConfig compute_kernel_config,
        bool enable_act_double_buffer,
        bool enable_weights_double_buffer,
        bool enable_split_reader,
        bool enable_subblock_padding) :
        output_channels(output_channels),
        groups(groups),
        sliding_window_config(sliding_window_config),
        untilize_out(untile_out),
        has_bias(has_bias),
        activation(activation),
        parallelization_config(p_config),
        block_config(b_config),
        memory_config(memory_config),
        dtype(dtype),
        input_tensor_shape(input_tensor_shape),
        compute_kernel_config(compute_kernel_config),
        enable_act_double_buffer(enable_act_double_buffer),
        enable_weights_double_buffer(enable_weights_double_buffer),
        enable_split_reader(enable_split_reader),
        enable_subblock_padding(enable_subblock_padding) {}

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    tt::tt_metal::operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "parallelization_config",
        "block_config",
        "sliding_window_config",
        "memory_config",
        "compute_kernel_config",
        "output_channels",
        "untilize_out",
        "has_bias",
        "activation",
        "dtype",
        "input_tensor_shape",
        "enable_act_double_buffer",
        "enable_weights_double_buffer",
        "enable_split_reader",
        "enable_subblock_padding");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->parallelization_config),
            std::cref(this->block_config),
            std::cref(this->sliding_window_config),
            std::cref(this->memory_config),
            std::cref(this->compute_kernel_config),
            std::cref(this->output_channels),
            std::cref(this->untilize_out),
            std::cref(this->has_bias),
            std::cref(this->activation),
            std::cref(this->dtype),
            std::cref(this->input_tensor_shape),
            std::cref(this->enable_act_double_buffer),
            std::cref(this->enable_weights_double_buffer),
            std::cref(this->enable_split_reader),
            std::cref(this->enable_subblock_padding));
    }
};

Tensor optimized_conv_new(
    const Tensor& a,
    const Tensor& b,
    std::optional<const Tensor> bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    const std::string& activation,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    const tt::tt_metal::MemoryConfig& memory_config,
    tt::tt_metal::DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool enable_act_double_buffer = false,
    bool enable_weights_double_buffer = false,
    bool enable_split_reader = false,
    bool enable_subblock_padding = false);

// Only enable packer l1 accumulation when there are in0_num_blocks_w > 2, otherwise
// unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
// does a spill and reload, so need more than 2 blocks to use l1 acc for packer
// For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
bool determine_packer_l1_acc(bool packer_l1_acc, bool enable_bias, uint32_t in0_num_blocks_w);

// Both CB and tensor allocation sizes are per per tensix core and in bytes.
struct conv_op_l1_usage {
    uint32_t tensor_allocation_size;
    uint32_t CB_allocation_size;
};

// This function calculates how much L1 will be allocated by the conv2d op.
// L1 allocation is either for the output tensor or for Circular Buffers.
// This doesn't include Circular Buffers that use globally allocated addresses, as these don't need memory allocation.
conv_op_l1_usage calculate_L1_usage(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const OptimizedConvBlockConfig& block_config,
    const OptimizedConvParallelizationConfig& pconfig,
    const ttnn::Shape& weights_shape,
    std::array<uint32_t, 2> kernel_size,
    const Conv2dConfig& conv_config,
    tt::tt_metal::DataType input_datatype,
    tt::tt_metal::DataType output_datatype,
    bool enable_bias,
    bool is_1d_depthwise_conv,
    bool skip_act_cb_create = false);

}  // namespace conv2d

}  // namespace operations::conv

}  // namespace ttnn

namespace optimized_conv_op_utils {
using namespace tt;
using namespace tt::tt_metal;

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(
    const ttnn::Shape& conv_activation_shape,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t num_cores_nhw,
    uint32_t act_block_h_ntiles);

}  // namespace optimized_conv_op_utils
