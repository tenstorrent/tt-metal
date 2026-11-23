// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "batch_norm_device_operation_types.hpp"
#include "batch_norm_program_factory.hpp"

namespace ttnn::prim::qsr {

struct BatchNormDeviceOperation {
    using operation_attributes_t = BatchNormParams;
    using tensor_args_t = BatchNormInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<BatchNormProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor batch_norm(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    float eps,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const ThreadPlan& threads = {});

}  // namespace ttnn::prim::qsr
