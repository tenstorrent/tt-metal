// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step2 {

struct MorehClipGradNormStep2Operation {
    struct operation_attributes_t {
        const float norm_type;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& tmp_pow_sum;
        const std::optional<Tensor>& total_norm;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& total_norm);

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step2

namespace ttnn::prim {
ttnn::operations::moreh::moreh_clip_grad_norm_step2::MorehClipGradNormStep2Operation::tensor_return_value_t
moreh_clip_grad_norm_step2(
    const Tensor& tmp_pow_sum,
    float norm_type,
    const std::optional<Tensor>& total_norm,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);
}  // namespace ttnn::prim
