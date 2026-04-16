// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_helper.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::moreh::moreh_nll_loss_step1 {

struct MorehNllLossStep1DeviceOperation {
    struct operation_attributes_t {
        const std::string reduction = NONE;
        const uint32_t ignore_index = std::numeric_limits<uint32_t>::max();
        const DataType dtype;
        const uint32_t channel_size = 1;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& target_tensor;
        const std::optional<Tensor>& weight_tensor;
    };

    using spec_return_value_t = TensorSpec;

    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step1

namespace ttnn::prim {
ttnn::operations::moreh::moreh_nll_loss_step1::MorehNllLossStep1DeviceOperation::tensor_return_value_t
moreh_nll_loss_step1(
    const Tensor& target_tensor,
    const std::optional<Tensor>& weight_tensor,
    int32_t ignore_index,
    const std::string& reduction,
    DataType dtype,
    uint32_t channel_size,
    const std::optional<MemoryConfig>& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config);
}  // namespace ttnn::prim
