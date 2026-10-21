// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "layernorm_device_operation_types.hpp"
#include "layernorm_welford_program_factory.hpp"

namespace ttnn::prim::qsr {

struct LayerNormDeviceOperation {
    using operation_attributes_t = LayerNormParams;
    using tensor_args_t = LayerNormInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<LayerNormWelfordProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const ttnn::prim::LayerNormProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<const Tensor>& recip_tensor = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& dtype = std::nullopt);

}  // namespace ttnn::prim::qsr
