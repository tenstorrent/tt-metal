// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/conv2d_new/device/factory/sharded_descriptor.hpp"
#include "ttnn/operations/conv2d_new/device/factory/width_sharded_descriptor.hpp"

namespace ttnn::prim {

// ---------------------------------------------------------------
// Conv2dNewDeviceOperation -- ProgramDescriptor variant of conv2d.
//
// Functionally identical to Conv2dDeviceOperation but both program
// factories use ProgramDescriptor for construction.
//
// Both factories are MeshWorkloadFactoryConcept because they need
// to create config tensors (sliding window indices) in
// create_mesh_workload, which cannot be done inside the pure
// create_descriptor function.
// ---------------------------------------------------------------
struct Conv2dNewDeviceOperation {
    using operation_attributes_t = Conv2dParams;
    using hashable_operation_attributes_t = Conv2dHashableParams;
    using tensor_args_t = Conv2dInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<
        conv2d_new_detail::Conv2dShardedDescriptorFactory,
        conv2d_new_detail::Conv2dWidthShardedDescriptorFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor);
};

}  // namespace ttnn::prim

namespace ttnn::prim {

// Prim-level entry point for Conv2dNew, taking the same prepared inputs
// as ttnn::prim::conv2d().
Tensor conv2d_new(
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
