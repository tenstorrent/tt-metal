// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>
#include <variant>

#include "deepseek_moe_gate_device_operation_types.hpp"
#include "deepseek_moe_gate_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate {

struct DeepseekMoeGateDeviceOperation {
    using operation_attributes_t =
        ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::operation_attributes_t;
    using tensor_args_t = ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::tensor_args_t;
    using tensor_return_value_t =
        ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::tensor_return_value_t;
    using spec_return_value_t = ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate::spec_return_value_t;
    using program_factory_t = std::variant<program::DeepseekMoeGateProgramFactory>;

    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& bias_tensor,
        const Tensor& input_indices_tensor,
        const Tensor& output_tensor,
        const Tensor& output_indices_tensor,
        float eps,
        float scaling_factor,
        bool enable_sigmoid);
};

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate
