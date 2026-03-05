// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "topk_router_gpt_device_operation_types.hpp"
#include "topk_router_gpt_program_factory.hpp"

namespace ttnn::operations::experimental::topk_router_gpt {

struct TopkRouterGptDeviceOperation {
    using operation_attributes_t = topk_router_gpt::operation_attributes_t;
    using tensor_args_t = topk_router_gpt::tensor_args_t;
    using tensor_return_value_t = topk_router_gpt::tensor_return_value_t;
    using spec_return_value_t = topk_router_gpt::spec_return_value_t;

    using program_factory_t = std::variant<TopkRouterGptProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        const Tensor& bias_tensor,
        const Tensor& output_tensor,
        uint32_t k,
        uint32_t num_experts,
        bool untilize_output,
        bool produce_hidden_rm);
};

}  // namespace ttnn::operations::experimental::topk_router_gpt

namespace ttnn::experimental {

constexpr auto topk_router_gpt = ttnn::register_operation<
    "ttnn::experimental::topk_router_gpt",
    ttnn::operations::experimental::topk_router_gpt::TopkRouterGptDeviceOperation>();

}  // namespace ttnn::experimental
