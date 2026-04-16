// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::moreh::moreh_group_norm_backward {
struct MorehGroupNormBackwardInputGradOperation {
    struct operation_attributes_t {
        const uint32_t num_groups;
        const MemoryConfig input_grad_memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };
    struct tensor_args_t {
        const Tensor& output_grad;
        const Tensor& input;
        const Tensor& mean;
        const Tensor& rstd;
        const std::optional<const Tensor> gamma;
        const std::optional<const Tensor> input_grad;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& outputs);

    static void validate_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward

namespace ttnn::prim {
ttnn::operations::moreh::moreh_group_norm_backward::MorehGroupNormBackwardInputGradOperation::tensor_return_value_t
moreh_group_norm_backward_input_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t num_groups,
    const std::optional<const Tensor>& gamma = std::nullopt,
    const std::optional<const Tensor>& input_grad = std::nullopt,
    const std::optional<MemoryConfig>& input_grad_memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
}  // namespace ttnn::prim
