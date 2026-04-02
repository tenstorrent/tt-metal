// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
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
        MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& scores;
        const Tensor& bias;
    };

    using spec_return_value_t = std::array<TensorSpec, 2>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            tt::tt_metal::KernelHandle compute_kernel_id{};
            std::vector<tt::tt_metal::CoreCoord> cores;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
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
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn::prim
