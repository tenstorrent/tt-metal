// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "moe_fused_swiglu_program_factory.hpp"
#include "moe_fused_swiglu_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {

struct MoeFusedSwiGluDeviceOperation {
    using operation_attributes_t = OperationArguments;
    using tensor_args_t = TensorArguments;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments);
    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments);
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_arguments,
        const tensor_args_t& tensor_arguments,
        tensor_return_value_t& output);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu

namespace ttnn::prim {

ttnn::Tensor moe_fused_swiglu(
    const ttnn::Tensor& activations,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    uint32_t m_tiles,
    uint32_t grid_x,
    uint32_t grid_y,
    bool read_x_at_offset,
    ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::RoutedExpertActivation activation,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& optional_output,
    const std::optional<ttnn::Tensor>& expert_region_offsets);

}  // namespace ttnn::prim
