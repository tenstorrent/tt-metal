// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op_width_sharded_program_factory.hpp"

#include <string>
#include <utility>
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"

namespace ttnn::prim {

namespace sliding_window = ttnn::operations::sliding_window;

struct Conv2dDeviceOperation {
    using operation_attributes_t = Conv2dParams;
    using hashable_operation_attributes_t = Conv2dHashableParams;
    using tensor_args_t = Conv2dInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<Conv2dShardedProgramFactory, Conv2dWidthShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor);
};

// Only enable packer l1 accumulation when there are in0_num_blocks_w > 2, otherwise
// unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
// does a spill and reload, so need more than 2 blocks to use l1 acc for packer
// For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
bool determine_packer_l1_acc(bool packer_l1_acc, bool enable_bias, uint32_t in0_num_blocks_w);

// L1 allocation is either for the output tensor or for Circular Buffers.
conv_op_l1_usage calculate_L1_usage(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const Conv2dBlockConfig& block_config,
    const Conv2dParallelizationConfig& pconfig,
    const ttnn::Shape& weights_shape,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    std::array<uint32_t, 2> dilation,
    const Conv2dConfig& conv_config,
    tt::tt_metal::DataType input_datatype,
    tt::tt_metal::DataType output_datatype,
    uint32_t output_image_width,
    bool enable_bias,
    bool is_1d_depthwise_conv,
    uint32_t input_channels_padded,
    bool skip_act_cb_create = false);

Tensor conv2d(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    const Conv2dParallelizationConfig& parallelization_config,
    const Conv2dBlockConfig& block_config,
    const tt::tt_metal::MemoryConfig& memory_config,
    tt::tt_metal::DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool full_inner_dim,
    bool enable_activation_reuse,
    bool config_tensors_in_dram,
    std::optional<bool> force_split_reader);

}  // namespace ttnn::prim
