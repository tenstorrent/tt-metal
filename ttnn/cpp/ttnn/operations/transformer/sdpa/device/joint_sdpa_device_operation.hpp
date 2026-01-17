// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_program_factory.hpp"

namespace ttnn::prim {

struct JointSDPADeviceOperation {
    using operation_attributes_t = JointSDPAParams;
    using tensor_args_t = JointSDPAInputs;
    using spec_return_value_t = JointSDPAResultSpec;
    using tensor_return_value_t = JointSDPAResult;
    using program_factory_t = std::variant<JointSDPAProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

JointSDPAResult joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::prim
