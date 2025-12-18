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

namespace ttnn::operations::transformer::sdpa::joint_sdpa {

struct JointSDPADeviceOperation {
    using operation_attributes_t = joint_sdpa::operation_attributes_t;
    using tensor_args_t = joint_sdpa::tensor_args_t;
    using spec_return_value_t = joint_sdpa::spec_return_value_t;
    using tensor_return_value_t = joint_sdpa::tensor_return_value_t;
    using program_factory_t = std::variant<program::JointSDPAProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const ttnn::Tensor& joint_tensor_q,
        const ttnn::Tensor& joint_tensor_k,
        const ttnn::Tensor& joint_tensor_v,
        const std::string& joint_strategy,
        const std::optional<SDPAProgramConfig>& program_config = std::nullopt,
        std::optional<float> scale = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::transformer::sdpa::joint_sdpa

namespace ttnn::prim {

constexpr auto joint_scaled_dot_product_attention = ttnn::register_operation<
    "ttnn::prim::joint_scaled_dot_product_attention",
    ttnn::operations::transformer::sdpa::joint_sdpa::JointSDPADeviceOperation>();

}  // namespace ttnn::prim
