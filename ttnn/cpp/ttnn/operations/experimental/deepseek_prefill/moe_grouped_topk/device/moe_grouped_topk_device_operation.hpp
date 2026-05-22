// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

struct MoeGroupedTopkDeviceOperation {
    struct operation_attributes_t {
        uint32_t n_groups;
        uint32_t summed_experts_per_group;
        uint32_t topk_groups;
        uint32_t n_activated_experts;
        float route_scale;
        float epsilon;
        bool stable_sort;
        tt::tt_metal::MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& scores;
        const Tensor& bias;
    };

    using spec_return_value_t = std::array<TensorSpec, 2>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::MoeGroupedTopkDeviceOperation::tensor_return_value_t
moe_grouped_topk(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    bool stable_sort = false,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn::prim
