// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "moe_gpt_fused_device_operation_types.hpp"
#include "moe_gpt_fused_program_factory.hpp"

namespace ttnn::operations::experimental::moe_gpt_fused {

struct MoEGPTFusedDeviceOperation {
    using operation_attributes_t = moe_gpt_fused::operation_attributes_t;
    using tensor_args_t = moe_gpt_fused::tensor_args_t;
    using spec_return_value_t = moe_gpt_fused::spec_return_value_t;
    using tensor_return_value_t = moe_gpt_fused::tensor_return_value_t;
    using program_factory_t = std::variant<program::MoEGPTFusedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& expert_indices,
        const Tensor& expert_scores,
        const Tensor& w0_w1_tensor,
        const Tensor& w2_tensor,
        uint32_t num_experts,
        uint32_t layer_id,
        uint32_t experts_per_device);
};

}  // namespace ttnn::operations::experimental::moe_gpt_fused

namespace ttnn::prim {
constexpr auto moe_gpt_fused = ttnn::register_operation<
    "ttnn::prim::moe_gpt_fused",
    ttnn::operations::experimental::moe_gpt_fused::MoEGPTFusedDeviceOperation>();
}  // namespace ttnn::prim
