// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <utility>
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"

namespace ttnn::operations::conv::conv2d {

using Conv2dSliceConfig = op_slicing::Op2DSliceConfig;
struct Conv2dConfig {
    // If set, the weights & bias tensors will be converted to this dtype after preprocessing.
    // prepare_conv_bias needs this to always be set to the same dtype as the weights.
    std::optional<tt::tt_metal::DataType> weights_dtype = std::nullopt;

    // Fused activation function as UnaryWithParam
    std::optional<ttnn::operations::unary::UnaryWithParam> activation = std::nullopt;

    // If user tensor will be deallocated if it's on device in L1 (will have no effect if input tensor is in DRAM).
    bool deallocate_activation = false;

    // If true && dellocate_activation is true, then after halo device op is done,
    // the output tensor of halo will be reallocated.
    bool reallocate_halo_output = true;

    // If true, config tensors for Conv2D are stored in DRAM instead of L1_SMALL. L1_SMALL is persistent storage and
    // get's quickly used up for large CNNs.
    bool config_tensors_in_dram = false;

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

    // used only if override_sharding_config or override_output_sharding_config is true
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

    // Applies only to block sharded layout.
    // By default inner dim of activation matrix will be sliced by kernel_h.
    // If L1 constraints allowed it we can use full inner dim.
    // This will increase perf, but it will take more L1 space.
    bool full_inner_dim = false;

    // ==================== EXPERIMENTAL FEATURES ====================
    // Features in this section are under development.
    // Use with caution.

    // Kernel Stride Folding (Issue: #22378)
    // Enables tensor folding optimization where:
    // - Input tensor (NHWC) is reshaped to (N, H/stride[0], W/stride[1], C * stride[0] * stride[1])
    // - Weight tensor (OC, IC, kernel[0], kernel[1]) is reshaped and permuted to (1, 1, IC * (kernel[0] + pad_h) *
    // (kernel[1] + pad_w), OC).
    //     Note: The zero padding applied to the weight tensor is implicit and not passed by the user via the padding
    //     argument, where pad_h = kernel[0] % stride[0] and pad_w = kernel[1] % stride[1].
    //
    // Note: This optimization is currently only applied when all of the following conditions are met:
    //    1. The input tensor is stored in DRAM memory.
    //    2. The input tensor's height and width are divisible by the stride dimensions.
    //    3. Stride values are equal to or less than the kernel dimensions.
    //    4. Input tensor's padding must be zero.
    //    5. Input tensor data type is not BFLOAT8_B.

    std::optional<bool> enable_kernel_stride_folding = std::nullopt;

    // Activation reuse is a feature that enables reusing data between consecutive image rows.
    // It can be enabled for height sharding only and boosts im2col performance,
    // so its meant to be used for reader-bound convolutions.
    bool enable_activation_reuse = false;

    // Force split reader overrides split_reader heuristic.
    // Split reader can be enabled for height sharding only and boosts im2col performance (if not weights reader bound),
    // so its meant to be used for reader-bound convolutions.
    // If not Height sharded and activation block size height is not greater than 32, then this is ignored.
    // If not set, then split reader heuristic is used to determine if it should be enabled.
    std::optional<bool> force_split_reader = std::nullopt;

    // override_output_sharding_config enables the user to specify the memory config of the output tensor
    // This impacts the core grid that executes matmul part of conv2d
    // Feature is currently supported only for BLOCK_SHARDED layout, without DRAM slicing
    // Additionally, NHW number of cores must match between input and output tensors
    bool override_output_sharding_config = false;
    // ===============================================================

    static constexpr auto attribute_names = std::make_tuple(
        "weights_dtype",
        "activation",
        "deallocate_activation",
        "reallocate_halo_output",
        "config_tensors_in_dram",
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
        "full_inner_dim",
        "enable_kernel_stride_folding",
        "enable_activation_reuse",
        "force_split_reader",
        "override_output_sharding_config");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->weights_dtype),
            std::cref(this->activation),
            std::cref(this->deallocate_activation),
            std::cref(this->reallocate_halo_output),
            std::cref(this->config_tensors_in_dram),
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
            std::cref(this->full_inner_dim),
            std::cref(this->enable_kernel_stride_folding),
            std::cref(this->enable_activation_reuse),
            std::cref(this->force_split_reader),
            std::cref(this->override_output_sharding_config));
    }
};

// TODO: Accept parallelization
enum class Conv2dOpParallelizationStrategy { MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE };

struct Conv2dParallelizationConfig {
    CoreCoord grid_size;  // (x,y)
    uint32_t num_cores_nhw = 1;
    uint32_t num_cores_c_in = 1;
    uint32_t num_cores_c_out = 1;
    uint32_t per_core_out_matrix_height_ntile = 1;
    uint32_t per_core_out_matrix_width_ntile = 1;

    CoreCoord get_grid_size() const { return this->grid_size; }
};

struct Conv2dBlockConfig {
    uint32_t act_block_h_ntiles;
    uint32_t act_block_w_ntiles;
    uint32_t out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles;
};

struct operation_attributes_t {
    sliding_window::SlidingWindowConfig sliding_window_config{};
    uint32_t output_channels = 0;
    uint32_t groups = 0;
    bool untilize_out = false;
    bool has_bias = false;
    std::optional<ttnn::operations::unary::UnaryWithParam> activation;
    Conv2dParallelizationConfig parallelization_config{};
    Conv2dBlockConfig block_config{};
    tt::tt_metal::MemoryConfig memory_config;
    tt::tt_metal::DataType dtype = tt::tt_metal::DataType::INVALID;
    std::array<std::uint32_t, 4> input_tensor_shape{};
    DeviceComputeKernelConfig compute_kernel_config;
    bool enable_act_double_buffer = false;
    bool enable_weights_double_buffer = false;
    bool full_inner_dim = false;
    bool enable_activation_reuse = false;
    bool config_tensors_in_dram = false;
    uint32_t pre_op_l1_allocation_size_bytes = 0;
    std::optional<bool> force_split_reader;
};

struct hashable_operation_attributes_t {
    sliding_window::SlidingWindowConfig sliding_window_config{};
    uint32_t output_channels = 0;
    bool untilize_out = false;
    bool has_bias = false;
    std::optional<ttnn::operations::unary::UnaryWithParam> activation;
    Conv2dParallelizationConfig parallelization_config{};
    Conv2dBlockConfig block_config{};
    tt::tt_metal::MemoryConfig memory_config;
    tt::tt_metal::DataType dtype = tt::tt_metal::DataType::INVALID;
    std::array<std::uint32_t, 4> input_tensor_shape{};
    DeviceComputeKernelConfig compute_kernel_config;
    bool enable_act_double_buffer = false;
    bool enable_weights_double_buffer = false;
    bool enable_activation_reuse = false;
    bool config_tensors_in_dram = false;
    std::optional<bool> force_split_reader;
};

struct tensor_args_t {
    Tensor a;
    Tensor b;
    std::optional<Tensor> bias;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

// Both CB and tensor allocation sizes are per per tensix core and in bytes.
struct conv_op_l1_usage {
    uint32_t tensor_allocation_size;
    uint32_t CB_allocation_size;
};

}  // namespace ttnn::operations::conv::conv2d
